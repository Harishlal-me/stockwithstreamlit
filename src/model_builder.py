import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from config import Config

def build_multi_task_model(input_shape=(60, 20)) -> Model:  # FIXED: (60, 20)
    """
    Multi-task LSTM: 60 days x 20 features → 4 outputs
    """
    inputs = Input(shape=input_shape, name='sequence_input')
    
    # LSTM layers (fixed dropout parameter)
    lstm1 = LSTM(Config.LSTM_UNITS_1, return_sequences=True)(inputs)
    lstm1 = Dropout(Config.DROPOUT_1)(lstm1)
    
    lstm2 = LSTM(Config.LSTM_UNITS_2)(lstm1)
    lstm2 = Dropout(Config.DROPOUT_1)(lstm2)
    
    # Dense layers
    dense1 = Dense(Config.DENSE_UNITS_1, activation='relu')(lstm2)
    dense1 = Dropout(Config.DROPOUT_2)(dense1)
    
    dense2 = Dense(Config.DENSE_UNITS_2, activation='relu')(dense1)
    dense2 = Dropout(Config.DROPOUT_3)(dense2)
    
    # 4 Outputs
    tomorrow_dir = Dense(1, activation='sigmoid', name='tomorrow_output')(dense2)
    week_dir = Dense(1, activation='sigmoid', name='week_output')(dense2)
    tomorrow_ret = Dense(1, name='tomorrow_return')(dense2)
    week_ret = Dense(1, name='week_return')(dense2)
    
    model = Model(inputs=inputs, outputs=[tomorrow_dir, week_dir, tomorrow_ret, week_ret])
    
    model.compile(
        optimizer='adam',
        loss={
            'tomorrow_output': 'binary_crossentropy',
            'week_output': 'binary_crossentropy',
            'tomorrow_return': 'mse',
            'week_return': 'mse'
        },
        loss_weights=[1.0, 1.2, 0.1, 0.1],  # FIXED: list format
        metrics={'tomorrow_output': 'accuracy', 'week_output': 'accuracy'}
    )
    
    model.summary()
    return model
