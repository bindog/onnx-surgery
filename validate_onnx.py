import argparse
import numpy as np
import onnxruntime

import cv2


def validate(model_path, image_path, batch_size=1, input_shape=(224, 224)):
    # input_shape (w, h)
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_shape).astype(np.float32)
    image = image / 255.0
    img_data = np.expand_dims(image, 0)
    img_data = np.transpose(img_data, [0, 3, 1, 2])
    x = np.repeat(img_data, batch_size, axis=0).astype(np.float32)
    try:
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 4
        model = onnxruntime.InferenceSession(model_path, sess_options=session_option)
        ort_inputs_name = model.get_inputs()[0].name
        ort_ouputs_names = [out.name for out in model.get_outputs()]
        ort_outs = model.run(ort_ouputs_names, {ort_inputs_name: x.astype('float32')})
        if len(ort_outs) > 1:
            outputs = tuple([np.array(out).astype("float32") for out in ort_outs])
            for output in outputs:
                print("one of output shape: ", output.shape)
            return outputs
        else:
            outputs = np.array(ort_outs[0]).astype("float32")
            print("output shape: ", outputs.shape)
            return outputs
    except Exception as e:
        print("validate error, check error message below:")
        print(str(e))
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="onnx validate")
    parser.add_argument("--model", default="", type=str, required=True)
    parser.add_argument("--image", default="", type=str, required=True)
    args = parser.parse_args()

    if validate(args.model, args.image) is not None:
        print("this onnx model seems ok")
    else:
        print("something wrong, please check your onnx model according to the error message...")
