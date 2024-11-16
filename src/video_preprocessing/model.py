from tensorflow import keras
from config import VideoClassificationConfig


class VideoClassifierModel:
    def __init__(self, config: VideoClassificationConfig, num_classes: int):
        self.config = config
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self) -> keras.Model:
        
        # Input layers
        frame_features_input = keras.Input((self.config.max_sequence_length, self.config.num_features))
        mask_input = keras.Input((self.config.max_sequence_length,), dtype='bool')
        
        # Bidirectional GRU layers
        x = keras.layers.Bidirectional(
            keras.layers.GRU(
                512,
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.3
            )
        )(frame_features_input, mask=mask_input)
        
        x = keras.layers.GRU(
            256,
            dropout=0.4,
            recurrent_dropout=0.4
        )(x)
        
        # Dense layers with regularization
        x = keras.layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.5)(x)
        
        x = keras.layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.4)(x)
        
        # Output layer
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = keras.Model(
            [frame_features_input, mask_input],
            outputs,
            name='video_classifier'
        )
        
        return model

    def compile_model(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def get_model(self) -> keras.Model:
        return self.model


# Test the model builder
# if __name__ == "__main__":
#     # Load configuration
#     config = VideoClassificationConfig()
#     num_classes = 10
    
#     # Build model
#     model_builder = VideoClassifierModel(config, num_classes)
#     model_builder.compile_model()
#     model = model_builder.get_model()
    
#     # Print model summary
#     print(model.summary())