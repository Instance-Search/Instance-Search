import numpy as np
import onnx
import onnxruntime as rt
import cv2
 
import onnx

def change_input_dim(model):
    # Use some symbolic name not used for any other dimension
    sym_batch_dim = "N"
    # or an actal value
    actual_batch_dim = 4

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = sym_batch_dim
#model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = sym_batch_dim


def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)
    return model

model = apply(change_input_dim, r"reid_resnet50_30.onnx", r"reid_resnet50_30_lb.onnx")

#create input data
imgpath = './21742_44030659001320010191_2684.jpg'
img = cv2.imread(imgpath)
img = cv2.resize(img,(128,384))
img = img[:,:,::-1]
mean = np.asarray([123.675,116.280,103.530])
std = np.asarray([57.0,57.0,57.0])
img = (img-mean)/std
img = img[np.newaxis,:,:,:]
img = img.transpose((0,3,1,2)).astype(np.float32)
print(img.shape)
imgs=np.vstack((img,img))
#print(img.shape)
#create runtime session
sess = rt.InferenceSession("reid_resnet50_30_lb.onnx")
# get output name
input_name = sess.get_inputs()[0].name
print("input name", input_name)
output_name= sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
#forward model
res = sess.run([output_name], {input_name: img})
ress = sess.run([output_name], {input_name: imgs})
feat = np.array(res)
feats = np.array(ress)
print(feat.shape)
print(feats.shape)

print(feat[0][0] - feats[0][1])
#f = open('./feat_onnx.txt','w')
#f.write(','.join(list(feat.flatten().astype(str))))
#f.close()

