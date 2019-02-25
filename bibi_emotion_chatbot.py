import bibi
import realtime_emotion_outline
import multiprocessing as mp

start = 0
end = 0
def abcdefg():
    while True:
        a = input('pleace input:')
        if a == 's':
            start = len(emotion_result) - 1
        elif a == 'e':
            end = len(emotion_result) -1
            print(emotion_result[start, end+1])


manager = mp.Manager()
emotion_result = manager.list()

bibi_chatbot = mp.Process(target=bibi.init)
predict_emotion = mp.Process(target=realtime_emotion_outline.predict_emotion, args=(emotion_result,))
print_emotion = mp.Process(target=abcdefg)

bibi_chatbot.start()
predict_emotion.start()
print_emotion.start()

bibi_chatbot.join()
predict_emotion.join()
print_emotion.join()


