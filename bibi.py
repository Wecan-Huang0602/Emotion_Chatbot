import Snowboy_thing.snowboydecoder as snowboydecoder
import signal

import speech_recognition
from gtts import gTTS
import pygame
import Jacky_Chatbot

import realtime_emotion_outline
import multiprocessing as mp
from collections import Counter

interrupted = False

txt_file = 'text_model/'

def get_emotion_result(start, end):
	result = list(emotion_result)[start:end + 1]
	# print('range: ', start, end)
	print('range_emotion: ', result)
	emotion_counts = Counter(result)
	final_emotion = emotion_counts.most_common(1)
	# print(final_emotion)
	print('情緒判斷結果：', final_emotion[0][0])

	global txt_file
	txt_file = 'text_model/' + (final_emotion[0][0] + '.txt')


def playsound():
	pygame.mixer.init()
	# 載入音檔
	soundWAV = pygame.mixer.Sound("Snowboy_thing/ding.wav")
	count = 0
	while True:
		# 檢查有沒有在播放音檔，如果沒有則開始播放
		if pygame.mixer.music.get_busy() == False:
			soundWAV.play()

			if pygame.mixer.music.set_endevent(pygame.USEREVENT) == None:
				count += 1
				if count > 1:
					pygame.mixer.music.stop()
					break


def stt():
	r = speech_recognition.Recognizer()

	with speech_recognition.Microphone() as source:
		audio = r.listen(source, phrase_time_limit=3)

	try:
		say_text = r.recognize_google(audio, language='zh-TW')
		return say_text
	except speech_recognition.UnknownValueError:
		return '發生錯誤'

def tts(text):
	tts = gTTS(text=text, lang='zh-tw')
	tts.save('save.mp3')

	pygame.mixer.init()
	#載入音檔
	pygame.mixer.music.load("save.mp3")
	count = 0
	while True:
		#檢查有沒有在播放音檔，如果沒有則開始播放
		if pygame.mixer.music.get_busy()==False:
			pygame.mixer.music.play()

			if pygame.mixer.music.set_endevent(pygame.USEREVENT) == None:
				count += 1
				if count > 1:
					pygame.mixer.music.stop()
					break

def chat():
	playsound()
	detector.terminate()
	chatbot = Jacky_Chatbot.Jacky_Chatbot()

	print('請開始說話～～～')

	start = len(emotion_result) - 1
	say_text = stt()
	end = len(emotion_result) - 1
	get_emotion_result(start, end)
	print('Q: ' + say_text)

	if say_text == '發生錯誤':
		print('A: 抱歉我沒聽清楚，請再說一次\n')
		tts('抱歉我沒聽清楚，請再說一次')
	else:
		chatbot.setFile(txt_file)
		response = chatbot.chat_with_chatbot(say_text)
		print('A: ' + response)
		tts(response)

	detector.start(detected_callback=chat,
				   interrupt_check=interrupt_callback,
				   sleep_time=0.03)


def signal_handler(signal, frame):
    global interrupted
    interrupted = True


def interrupt_callback():
    global interrupted
    return interrupted


detector = None
def init():
	model = 'Snowboy_thing/小白.pmdl'

	# capture SIGINT signal, e.g., Ctrl+C
	signal.signal(signal.SIGINT, signal_handler)

	global detector
	detector = snowboydecoder.HotwordDetector(model, sensitivity=0.5)
	print('Listening... Press Ctrl+C to exit')


	detector.start(detected_callback=chat,
				   interrupt_check=interrupt_callback,
				   sleep_time=0.03)








manager = mp.Manager()
emotion_result = manager.list()

bibi_chatbot = mp.Process(target=init)
predict_emotion = mp.Process(target=realtime_emotion_outline.predict_emotion, args=(emotion_result,))

bibi_chatbot.start()
predict_emotion.start()

bibi_chatbot.join()
predict_emotion.join()