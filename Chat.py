import speech_recognition
from gtts import gTTS
import pygame
import Jacky_Chatbot

def stt():
	r = speech_recognition.Recognizer()

	with speech_recognition.Microphone() as source:
		audio = r.listen(source)

	say_text = r.recognize_google(audio, language='zh-TW')
	return say_text

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



chatbot = Jacky_Chatbot.Jacky_Chatbot()
chatbot.setFile('Happy.txt')

while True:
	text = input('請按Enter鍵：')
	if text == 'q':
		break
	print('請開始說話～～～')

	say_text = stt()
	print('Q: ' + say_text)

	response = chatbot.chat_with_chatbot(say_text)
	print('A: ' + response)
	tts(response)