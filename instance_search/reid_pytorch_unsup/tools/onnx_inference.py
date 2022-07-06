import numpy as np
import onnx
import onnxruntime as rt
import cv2
 
#create input data
imgpath = './0108_03_0000.png'
img = cv2.imread(imgpath)
img = cv2.resize(img,(128,384))
img = img[:,:,::-1]
mean = np.asarray([123.675,116.280,103.530])
std = np.asarray([57.0,57.0,57.0])
img = (img-mean)/std
img = img[np.newaxis,:,:,:]
img = img.transpose((0,3,1,2)).astype(np.float32)
#create runtime session
sess = rt.InferenceSession("")
# get output name
input_name = sess.get_inputs()[0].name
print("input name", input_name)
output_name= sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
#forward model
res = sess.run([output_name], {input_name: img})
feat = np.array(res)
f = open('./feat_onnx.txt','w')
f.write(','.join(list(feat.flatten().astype(str))))
f.close()
