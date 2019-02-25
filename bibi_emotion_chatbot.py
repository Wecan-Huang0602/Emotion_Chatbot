import bibi
import realtime_emotion_outline
import multiprocessing as mp

manager = mp.Manager()
emotion_result = manager.dict()

bibi_chatbot = mp.Process(target=bibi.init)
bibi_chatbot.start()

predict_emotion = mp.Process(target=realtime_emotion_outline.predict_emotion, args=(1, emotion_result))
predict_emotion.start()

