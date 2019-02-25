import cv2

def detect(img, detector, fa):
    faces = detector(img, 1)
    if len(faces) == 0:
        return img

    for face in faces:
        faceAligned = fa.align(img, img, face)

    return faceAligned


def draw_face(img, detector, detector_68landmark, fa):
  img = detect(img, detector, fa)

  scale = img.shape[1] / img.shape[0]
  img = cv2.resize(img, (int(300*scale), 300), interpolation=cv2.INTER_CUBIC)

  x_1 = 0
  y_1 = 0
  x_2 = 0
  y_2 = 0

  # 偵測人臉
  face_rects, scores, idx = detector.run(img, 0)

  # 取出所有偵測的結果
  for i, d in enumerate(face_rects):
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()

    # 68點landmark
    landmark_shape = detector_68landmark(img, d)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # 左眉毛
    for i in range(17, 21):
      cv2.line(img, (landmark_shape.part(i).x, landmark_shape.part(i).y),
               (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)

    # 右眉毛
    for i in range(22, 26):
      cv2.line(img, (landmark_shape.part(i).x, landmark_shape.part(i).y),
               (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)

    # 左眼
    for i in range(36, 41):
      cv2.line(img, (landmark_shape.part(i).x, landmark_shape.part(i).y),
               (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)
    cv2.line(img, (landmark_shape.part(41).x, landmark_shape.part(41).y),
             (landmark_shape.part(36).x, landmark_shape.part(36).y), (255, 255, 255), 1)

    # 右眼
    for i in range(42, 47):
      cv2.line(img, (landmark_shape.part(i).x, landmark_shape.part(i).y),
               (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)
    cv2.line(img, (landmark_shape.part(47).x, landmark_shape.part(47).y),
             (landmark_shape.part(42).x, landmark_shape.part(42).y), (255, 255, 255), 1)

    # 鼻子
    for i in range(27, 30):
      cv2.line(img, (landmark_shape.part(i).x, landmark_shape.part(i).y),
               (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)
    for i in range(31, 35):
      cv2.line(img, (landmark_shape.part(i).x, landmark_shape.part(i).y),
               (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)

    # 嘴巴
    for i in range(48, 59):
      cv2.line(img, (landmark_shape.part(i).x, landmark_shape.part(i).y),
               (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)
    cv2.line(img, (landmark_shape.part(59).x, landmark_shape.part(59).y),
             (landmark_shape.part(48).x, landmark_shape.part(48).y), (255, 255, 255), 1)
    for i in range(60, 67):
      cv2.line(img, (landmark_shape.part(i).x, landmark_shape.part(i).y),
               (landmark_shape.part(i + 1).x, landmark_shape.part(i + 1).y), (255, 255, 255), 1)
    cv2.line(img, (landmark_shape.part(67).x, landmark_shape.part(67).y),
             (landmark_shape.part(60).x, landmark_shape.part(60).y), (255, 255, 255), 1)

    x_1 = x1
    x_2 = x2
    y_1 = y1
    y_2 = y2

  img = img[y_1:y_2, x_1:x_2]

  cv2.imwrite('cut.jpg', img)
  return img
