from ultralytics import YOLO
import os, shutil
import numpy as np
import cv2, csv
import pandas as pd
from PIL import Image

def singleScaleRetinex(img, variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex
def multiScaleRetinex(img, variance_list):
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += singleScaleRetinex(img, variance)
    retinex = retinex / len(variance_list)
    return retinex
def MSR(img, variance_list):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, variance_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)
    return img_retinex

def run(path):
    model1 = YOLO('best.pt')
    model2 = YOLO('d_best.pt')
    file_path = path
    #folder_path = path
    folder_path_2 = "runs/detect/predict/crops/0/"
    #detected_files = []
    filess = []
    reading = {}

    results1 = model1.predict(file_path, save_crop=True, conf=0.5)

    variance_list = [20, 85, 35]  # [15, 80, 30]
    for f2 in os.listdir(folder_path_2):
        #detected_files.append(f2)
        file_path2 = os.path.join(folder_path_2, f2)
        image = cv2.imread(file_path2)
        img_msr = MSR(image, variance_list)
        cv2.imwrite(file_path2, img_msr)

    results2 = model2.predict(file_path2, conf=0.2)
    prediction = ''
    dummy = ''
    for r in results2:
        data = np.array(r.boxes.data)

    sorted_bounding_boxes = sorted(data, key=lambda x: x[0])

    for box in sorted_bounding_boxes:
        prediction = prediction + str(int(box[5]))

    #files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Print the paths of all files
    # for file in files:
    #     file_path = os.path.join(folder_path, file)
    #     results1 = model1.predict(file_path, save_crop=True, conf=0.5)

    # for f in os.listdir(folder_path):
    #     filess.append(f)



    # for file2 in detected_files:
    #     file_path2 = os.path.join(folder_path_2, file2)
    #     results2 = model2.predict(file_path2, conf=0.4)
    #     prediction = ''
    #     dummy = ''
    #     for r in results2:
    #         data = np.array(r.boxes.data)
    #
    #     sorted_bounding_boxes = sorted(data, key=lambda x: x[0])
    #
    #     for box in sorted_bounding_boxes:
    #         prediction = prediction + str(int(box[5]))

        # if (len(prediction) == 5):
        #     reading[str(file2)] = prediction
        #
        # if(len(prediction)>5):
        #     for i in range(5):
        #         dummy=dummy+prediction[i]
        #     reading[str(file2)]=dummy
        # if (prediction == ''):
        #     reading[str(file2)] = "Unable to detect"
        # else:
        #     reading[str(file2)] = prediction

    # if not os.path.exists(destination_folder):
    #     os.makedirs(destination_folder)

    # common_elements = set(filess) & set(reading.keys())
    #
    # files_to_move = set(filess) - common_elements
    # for i in files_to_move:
    #     reading[i] = "Unable to detect"
    #     # source_path = os.path.join(folder_path, i)
    #     # shutil.move(source_path, destination_folder)
    #
    # file = 'results_18.txt'
    # output_data = {'Meters': list(reading.keys()), 'Readings': list(reading.values())}
    # print(output_data)
    # # df = pd.DataFrame(output_data)
    # # df.to_csv(file,sep ='\t',index=True)
    shutil.rmtree("runs/detect/predict/")
    return prediction


if __name__ == "__main__":
    dict = []
    path = 'D:/Comparition/'
    for f in os.listdir(path):
        filename = os.path.join(path, f)
        reading = run(filename)
        print(reading)
        dict.append(reading)


    print(dict)
    #df.to_csv('New_res')