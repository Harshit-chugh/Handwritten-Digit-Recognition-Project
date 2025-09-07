import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model.h5')
def sketch_recognition(img):
  x = model.predict(img.reshape(-1, 28, 28, 1)).tolist()[0]
  return {str(i):x[i] for i in range(10)}

interface = gr.Interface(fn=sketch_recognition, inputs="sketchpad",outputs=gr.outputs.Label(num_top_classes=3))
interface.launch()