import dlib
import cv2
import draw_face_class_outline
import tensorflow as tf
import model

from imutils.face_utils import  FaceAligner



class_list = ['Sad', 'Happy', 'Natural', 'Angry', 'Fear']

# 紀錄已經拍了幾張照片
count = 0

def readImg(file_dir):
    # 要讀的檔案的名稱
    image_name_list = [file_dir]

    # 產生一個檔案名稱的佇列
    filename_queue = tf.train.string_input_producer(image_name_list, shuffle=False)

    # 產生一個 reader，並用它來從檔案名稱佇列中讀取資料
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_jpeg(value, channels=3)  # 解碼

    # 調整圖片的大小
    image = tf.image.resize_images(image, (120, 120), method=0)
    # 把圖片標準化
    image = tf.image.per_image_standardization(image)
    # 把圖片變成四維的張量
    img_reshape = tf.reshape(image, shape=[1, 120, 120, 3])

    return img_reshape


def predict_emotion(emotion_result):
    # 使用電腦的視訊鏡頭
    cap = cv2.VideoCapture(0)

    # Dlib 的人臉偵測器
    detector = dlib.get_frontal_face_detector()

    # Dlib 68 點 landmark
    detector_68landmark = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # 人臉矯正器
    fa = FaceAligner(detector_68landmark, desiredFaceWidth=256)


    image = readImg('cut.jpg')
    sess = tf.Session()
    y_predict = model.inference(images=image,
                                batchSize=1,
                                nClasses=5)

    # 在 tf.train.string_input_producer 中定義了一個 num_epochs 變數，要對他進行初始化
    tf.local_variables_initializer().run(session=sess)
    # 使用 start_queue_runners 之後才會開始填充佇列
    threads = tf.train.start_queue_runners(sess=sess)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('model'))  # 載入模型參數


    # 以迴圈從影片檔案讀取影格，並顯示出來
    while(cap.isOpened()):
      ret, frame = cap.read()

      #   讓畫面可以鏡像顯示
      frame = cv2.flip(frame, 1, 0)


      # 將每一針的畫面轉成人臉特徵圖
      img = draw_face_class_outline.draw_face(img=frame,
                                      detector=detector,
                                      detector_68landmark=detector_68landmark,
                                      fa=fa)

      global count
      count += 1

      if img.shape[0] == 0:
        print('error')
      else:
        prediction_result = sess.run(tf.arg_max(y_predict, 1))
        confident_value = sess.run(tf.math.reduce_max(tf.nn.softmax(y_predict)))

        emotion_result.append(class_list[int(prediction_result[0])])
        # print('==========\033[0;32m(%d)\033[0m==========' % count)
        # print('Presict: ' + class_list[int(prediction_result[0])])
        # print('Confidence: ' + str(confident_value) + '\n')


      # 偵測人臉
      face_rects, scores, idx = detector.run(frame, 0)

      # 取出所有偵測的結果
      for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()

        confident_value_str = "   %1.2f" % float(confident_value)


        # 標示分數
        cv2.putText(frame, str(class_list[int(prediction_result[0])])+confident_value_str, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
                0.7, (255, 255, 255), 1, cv2.LINE_AA)


        # 68點landmark
        landmark_shape = detector_68landmark(frame, d)

        # 左眉毛
        for i in range(17, 21):
            cv2.line(frame, (landmark_shape.part(i).x, landmark_shape.part(i).y),
                     (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)

        # 右眉毛
        for i in range(22, 26):
            cv2.line(frame, (landmark_shape.part(i).x, landmark_shape.part(i).y),
                     (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)

        # 左眼
        for i in range(36, 41):
            cv2.line(frame, (landmark_shape.part(i).x, landmark_shape.part(i).y),
                     (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)
        cv2.line(frame, (landmark_shape.part(41).x, landmark_shape.part(41).y),
                 (landmark_shape.part(36).x, landmark_shape.part(36).y), (255, 255, 255), 1)

        # 右眼
        for i in range(42, 47):
            cv2.line(frame, (landmark_shape.part(i).x, landmark_shape.part(i).y),
                     (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)
        cv2.line(frame, (landmark_shape.part(47).x, landmark_shape.part(47).y),
                 (landmark_shape.part(42).x, landmark_shape.part(42).y), (255, 255, 255), 1)

        # 鼻子
        for i in range(27, 30):
            cv2.line(frame, (landmark_shape.part(i).x, landmark_shape.part(i).y),
                     (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)
        for i in range(31, 35):
            cv2.line(frame, (landmark_shape.part(i).x, landmark_shape.part(i).y),
                     (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)

        # 嘴巴
        for i in range(48, 59):
            cv2.line(frame, (landmark_shape.part(i).x, landmark_shape.part(i).y),
                     (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)
        cv2.line(frame, (landmark_shape.part(59).x, landmark_shape.part(59).y),
                 (landmark_shape.part(48).x, landmark_shape.part(48).y), (255, 255, 255), 1)

        for i in range(60, 67):
            cv2.line(frame, (landmark_shape.part(i).x, landmark_shape.part(i).y),
                     (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)
        cv2.line(frame, (landmark_shape.part(67).x, landmark_shape.part(67).y),
                 (landmark_shape.part(60).x, landmark_shape.part(60).y), (255, 255, 255), 1)

        # 以方框標示偵測的人臉
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)



      # 顯示結果
      cv2.imshow("Face Detection", frame)

      # ord() 可以取得字元的ACSII碼
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()
