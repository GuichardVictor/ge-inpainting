import keras
from keras.models import Model
from keras.layers import Input, Dropout, LeakyReLU, Activation, Lambda
from keras.layers import Conv3D, UpSampling3D, BatchNormalization
from keras.layers.merge import Concatenate
from keras.applications import VGG16
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

from ..layers import PConv3D

from keras.applications import VGG16


class PConvModel3D:

    def __init__(self, img_shape=(16, 256, 256, 1), many2many=True):
        """
        Creates a UNet like model with Partial Convolutions
        and VGG loss.
        """
        self.shape_3d = img_shape
        self.shape = img_shape[1:]

        # VGG Preprocessing settings
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.vgg_layers = [3, 6, 10] # taken from: https://arxiv.org/pdf/1804.07723.pdf
        
        # 256 >= ?
        # skipping assertions

        self.many2many = many2many

        self.vgg = self._create_vgg()
        self.model, self.masks = self._create_model(self.shape_3d)
        self._compile_model(self.model, self.masks)
    
    def fit_generator(self, generator, *args, **kwargs):
        self.model.fit_generator(
            generator,
            *args, **kwargs
        )
            
    def summary(self):
        print(self.model.summary())
    
    def predict(self, sample, **kwargs):
        """Run prediction using this model"""
        return self.model.predict(sample, **kwargs)
    
    def freeze_bn(self, count=3):
        """ Freezes batch normalization of the encoder as BatchNormalization has trouble with holes """
        for i in range(1, count+1):
            self.model.get_layer(f'EncodingBatchNorm{i}').trainable = False
        
        # Recompiling Model
        self._compile_model(self.model, self.masks)


    def _create_vgg(self):
        inputs = Input(shape=self.shape)

        preprocessing = Lambda(lambda x: (x - self.mean) / self.std)(inputs)

        vgg_model = VGG16(weights='imagenet', include_top=False)
        vgg_model.outputs = [vgg_model.layers[i].output for i in self.vgg_layers]
        
        model = Model(inputs=inputs, outputs=vgg_model(preprocessing))
        model.trainable = False
        model.compile(loss='mse', optimizer='adam')

        return model

    def _create_model(self, input_shape):
        def _encoding_layer(img_in, mask_in, filters, kernel_size, norm=True, train_norm=True):
            conv, mask = PConv3D(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])

            if norm: # Giving a name to batchnormalization as in paper encoding batch normalization are set to training = false in second train
                conv = BatchNormalization(name='EncodingBatchNorm'+str(_encoding_layer.counter))(conv, training=train_norm)
            
            conv = Activation('relu')(conv)
            _encoding_layer.counter += 1
            return conv, mask
        
        _encoding_layer.counter = 0

        def _decoding_layer(img_in, mask_in, enc_img, enc_mask, filters, kernel_size, norm=True):
            up_img = UpSampling3D(size=(2, 2, 2))(img_in)
            up_mask = UpSampling3D(size=(2, 2, 2))(mask_in)

            concat_img = Concatenate(axis=4)([enc_img, up_img])
            concat_mask = Concatenate(axis=4)([enc_mask, up_mask])

            conv, mask = PConv3D(filters, kernel_size, padding='same')([concat_img, concat_mask])
            if norm:
                conv = BatchNormalization()(conv)
            conv = LeakyReLU(alpha=0.2)(conv)
            return conv, mask
        
        def _backend_reshape(x):
            return K.reshape(x, (-1, *self.shape))

        
        inputs_img = Input(input_shape, name='inputs_img')
        inputs_mask = Input(input_shape, name='inputs_mask')

        e_conv1, e_mask1 = _encoding_layer(inputs_img, inputs_mask, 32, 7, norm=False)
        e_conv2, e_mask2 = _encoding_layer(e_conv1, e_mask1, 64, 5)
        e_conv3, e_mask3 = _encoding_layer(e_conv2, e_mask2, 128, 5)
        e_conv4, e_mask4 = _encoding_layer(e_conv3, e_mask3, 256, 3)

        d_conv5, d_mask5 = _decoding_layer(e_conv4, e_mask4, e_conv3, e_mask3, 256, 3)
        d_conv6, d_mask6 = _decoding_layer(d_conv5, d_mask5, e_conv2, e_mask2, 128, 3)
        d_conv7, d_mask7 = _decoding_layer(d_conv6, d_mask6, e_conv1, e_mask1, 64, 3)
        d_conv8, d_mask8 = _decoding_layer(d_conv7, d_mask7, inputs_img, inputs_mask, input_shape[-1], 3, norm=False)

        outputs = Conv3D(1, 1, activation = 'sigmoid', name='outputs_img')(d_conv8)

        if not self.many2many:
            outputs = Lambda(lambda x: x[:, input_shape[0] // 2])(outputs)
        
        model = Model(inputs=[inputs_img, inputs_mask], outputs=outputs)
        
        if self.many2many:
            mask_loss_layer = Lambda(_backend_reshape)(inputs_mask) 
        else:
            mask_loss_layer = Lambda(lambda x: x[:, input_shape[0] // 2])(inputs_mask)        

        return model, mask_loss_layer
    
    def _compile_model(self, model, masks, lr=0.0002):
        custom_loss = self._vgg_custom_loss_many2many(masks) if self.many2many else self._vgg_custom_loss_many2one(masks)
        model.compile(optimizer=Adam(learning_rate=lr), loss=custom_loss, metrics=[self.PSNR, self.SSIM])
    
    def _vgg_custom_loss_many2many(self, mask):
        # Most losses are adaptation from: https://keras.io/examples/neural_style_transfer/
        def loss(y_true, y_pred):
            
            y_true = tf.reshape(y_true, (-1, *(self.shape)))
            y_pred = tf.reshape(y_pred, (-1, *(self.shape)))

            y_comp = mask * y_true + (1-mask) * y_pred # only mask computation
            
            vgg_pred = self.vgg(y_pred)
            vgg_gt = self.vgg(y_true)
            vgg_comp = self.vgg(y_comp)

            # Loss components from: https://arxiv.org/pdf/1804.07723.pdf
            l1 = self.loss_valid(mask, y_true, y_pred)
            l2 = self.loss_hole(mask, y_true, y_pred)
            l3 = self.loss_perceptual(vgg_pred, vgg_gt, vgg_comp)
            l4 = self.loss_style(vgg_pred, vgg_gt)
            l5 = self.loss_style(vgg_comp, vgg_gt)
            l6 = self.loss_tv(mask, y_comp)

            return l1 + 6 * l2 + 0.05 * l3 + 120 * (l4+l5) + 0.1 * l6
        
        return loss

    def _vgg_custom_loss_many2one(self, mask):
        # Most losses are adaptation from: https://keras.io/examples/neural_style_transfer/
        def loss(y_true, y_pred):

            y_comp = mask * y_true + (1-mask) * y_pred # only mask computation
            
            vgg_pred = self.vgg(y_pred)
            vgg_gt = self.vgg(y_true)
            vgg_comp = self.vgg(y_comp)

            # Loss components from: https://arxiv.org/pdf/1804.07723.pdf
            l1 = self.loss_valid(mask, y_true, y_pred)
            l2 = self.loss_hole(mask, y_true, y_pred)
            l3 = self.loss_perceptual(vgg_pred, vgg_gt, vgg_comp)
            l4 = self.loss_style(vgg_pred, vgg_gt)
            l5 = self.loss_style(vgg_comp, vgg_gt)
            l6 = self.loss_tv(mask, y_comp)

            return l1 + 6 * l2 + 0.05 * l3 + 120 * (l4+l5) + 0.1 * l6
        
        return loss

    def loss_hole(self, mask, y_true, y_pred):
        """L1 Loss on content inside holes"""
        return self.l1((1-mask) * y_true, (1-mask) * y_pred)
    
    def loss_valid(self, mask, y_true, y_pred):
        """L1 loss on content outside holes"""
        return self.l1(mask * y_true, mask * y_pred)
    
    def loss_perceptual(self, vgg_out, vgg_gt, vgg_comp): 
        """Perceptual loss based on VGG16"""       
        loss = 0
        for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
            loss += self.l1(o, g) + self.l1(c, g)
        return loss
        
    def loss_style(self, output, vgg_gt):
        """Style loss"""
        loss = 0
        for o, g in zip(output, vgg_gt):
            loss += self.l1(self.matrix_gram(o), self.matrix_gram(g))
        return loss
    
    def loss_tv(self, mask, y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""

        # Create dilated hole region using a 3x3 kernel of all 1s.
        kernel = K.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
        #dilated_mask = K.conv2d(1-mask, kernel, data_format='channels_last', padding='same')

        # Cast values to be [0., 1.], and compute dilated hole region of y_comp
        dilated_mask = K.cast(K.greater(1 - mask, 0), 'float32')
        P = dilated_mask * y_comp

        # Calculate total variation loss
        a = self.l1(P[:,1:,:,:], P[:,:-1,:,:])
        b = self.l1(P[:,:,1:,:], P[:,:,:-1,:])        
        return a+b

    @staticmethod
    def PSNR(y_true, y_pred):
        """
        PSNR (Peek Signal to Noise Ratio): https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        """        
        return - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0) 
    
    @staticmethod
    def SSIM(y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

    @staticmethod
    def l1(y_true, y_pred):
        """ Simple Custom L1 Loss """
        if K.ndim(y_true) in [4, 3]:
            return K.mean(K.abs(y_pred - y_true), axis=[*range(1, K.ndim(y_true))])
        else:
            raise NotImplementedError("Model working on Images not 1D tensors.")
    
    @staticmethod
    def matrix_gram(x):
        assert K.image_data_format() == 'channels_last', 'gram computation was made with channel last only'
        assert K.ndim(x) == 4, 'Dimension is supposed to be: (batch_size, width, height, channels)'

        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3] # Unpacking does not work with tensors

        features = K.reshape(x, K.stack([B, C, H*W]))
        gram = K.batch_dot(features, features, axes=2)
        
        # Normalize with channels, height and width
        gram = gram /  K.cast(C * H * W, x.dtype)
        
        return gram