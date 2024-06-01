

# Python program to translate
# speech to text and text to speech
 
 
import speech_recognition as sr

from subprocess import Popen, PIPE
from time import sleep, perf_counter
from datetime import datetime
import board
import digitalio
import adafruit_character_lcd.character_lcd as characterlcd
import SentimentAnalysis
import numpy as np

# Initialize the recognizer 
r = sr.Recognizer() 
 



# Modify this if you have a different sized character LCD
lcd_columns = 16
lcd_rows = 2

# compatible with all versions of RPI as of Jan. 2019
# v1 - v3B+
lcd_rs = digitalio.DigitalInOut(board.D22)
lcd_en = digitalio.DigitalInOut(board.D17)
lcd_d4 = digitalio.DigitalInOut(board.D25)
lcd_d5 = digitalio.DigitalInOut(board.D24)
lcd_d6 = digitalio.DigitalInOut(board.D23)
lcd_d7 = digitalio.DigitalInOut(board.D18)


# Initialise the lcd class
lcd = characterlcd.Character_LCD_Mono(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6,
                                    lcd_d7, lcd_columns, lcd_rows)

# looking for an active Ethernet or WiFi device


# wipe LCD screen before we start
lcd.clear()


def display(str1, str2 = "", time = 5):
    lcd.message = str1 + "           \n" + str2 + ""
    sleep(time)
    lcd.message = "Listening...     \n"
 
def display_sentiment(text,prob):
    if prob > 0.5:
        display(text, "Positive: {:.2f}%".format(prob*100))
    else:
        display(text, "Negative: {:.2f}%".format((1-prob)*100))

# Loop infinitely for user to
# speak 
lcd.message = "Starting..."


#model_size = input("Model Size: [L]arge/[S]mall")

model_size = 'S'

if (model_size == 'L'):
    vocab_size = 850171 # +1 for the 0 padding
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2
    model = SentimentAnalysis.SentimentAnalysisModel(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, 0.5, True)
elif (model_size == 'S'):
    vocab_size = 66123 # +1 for the 0 padding
    output_size = 1
    embedding_dim = 400
    hidden_dim = 128
    n_layers = 2
    model = SentimentAnalysis.SentimentAnalysisModel(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, 0.5, False)
else:
    exit()

mode_txt = input("Usage Mode: [T]ext/[A]udio  ")

if (mode_txt != 'A' and mode_txt != 'T'):
    exit()

print("Ready")
display("Ready!         ", "", 2)
lcd.message = "Listening...        \n"

if (mode_txt == 'T'):
    while True:
        txt = input()
        display_sentiment(txt, model.predict_text(txt))

elif (mode_txt == 'A'):
    while(1):    
        
        # Exception handling to handle
        # exceptions at the runtime
        try:
            
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                
                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level 
                r.adjust_for_ambient_noise(source2, duration=0.2)
                
                #listens for the user's input 
                audio2 = r.listen(source2,10)
                
                # recognize audio

                #MyText = r.recognize_sphinx(audio2)
                MyText = r.recognize_google(audio2)
                
                MyText = MyText.lower()
    
                display_sentiment(MyText, model.predict_text(MyText))
                
        except sr.RequestError as e:
            display("Request Error")
            
        except sr.UnknownValueError:
            display("Not Recognized", "", 2.5)

        except sr.WaitTimeoutError:
            display("No Input Detected")
