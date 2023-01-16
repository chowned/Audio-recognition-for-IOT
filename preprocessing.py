import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
LABELS = ['change languagenone', 'activatemusic', 'deactivatelights', 'increasevolume', 'decreasevolume', 'increaseheat', 'decreaseheat', 'nannan']
# This is the file genarated that has the Labels that i must use for training
 
frame_length_in_s = 0.032
frame_step_in_s  = 0.032


PREPROCESSING_ARGS = {
    'downsampling_rate': 16000,
    'frame_length_in_s': frame_length_in_s,
    'frame_step_in_s': frame_step_in_s,
}

TRAINING_ARGS = {
    'batch_size': 32,
    'initial_learning_rate': 0.03,
    'end_learning_rate': 0.001, #1.e-9,
    'epochs': 1
}


final_sparsity = 0.01
alpha=1

num_mel_bins = (int) ((16000 - 16000 * PREPROCESSING_ARGS['frame_length_in_s'])/(16000*PREPROCESSING_ARGS['frame_step_in_s']))+1
print(num_mel_bins)

PREPROCESSING_ARGS = {
    **PREPROCESSING_ARGS,
    'num_mel_bins': num_mel_bins,
    'lower_frequency': 80,
    'upper_frequency': 8000,
}

downsampling_rate = PREPROCESSING_ARGS['downsampling_rate']
sampling_rate_int64 = tf.cast(downsampling_rate, tf.int64)
frame_length = int(downsampling_rate * PREPROCESSING_ARGS['frame_length_in_s'])
print("Frame_length: {}".format(frame_length))
frame_step = int(downsampling_rate * PREPROCESSING_ARGS['frame_step_in_s'])
print("Frame_length: {}".format(frame_step))
num_spectrogram_bins = frame_length // 2 + 1
num_mel_bins = PREPROCESSING_ARGS['num_mel_bins']
lower_frequency = PREPROCESSING_ARGS['lower_frequency']
upper_frequency = PREPROCESSING_ARGS['upper_frequency']

linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=num_mel_bins,
    num_spectrogram_bins=num_spectrogram_bins,
    sample_rate=downsampling_rate,
    lower_edge_hertz=lower_frequency,
    upper_edge_hertz=upper_frequency
)

def preprocess(filename):
    audio_binary = tf.io.read_file(filename)

    path_parts = tf.strings.split(filename, '_')
    path_end = path_parts[-1]
    file_parts = tf.strings.split(path_end, '.')
    true_label = file_parts[0]
    label_id = tf.argmax(true_label == LABELS)
    audio, sampling_rate = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1) #all our audio are mono, drop extra axis
    audio_padded = audio
    stft = tf.signal.stft(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)
    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, -1)  # channel axis
    mfcss = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    return mfcss, label_id
