

# Python program to translate
# speech to text and text to speech
 
 
import speech_recognition as sr

from subprocess import Popen, PIPE
from time import sleep, perf_counter
from datetime import datetime
import board
import digitalio
import adafruit_character_lcd.character_lcd as characterlcd


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



 
# Loop infinitely for user to
# speak 
lcd.message = "Listening..."

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
            audio2 = r.listen(source2)
             
            # recognize audio

            #MyText = r.recognize_sphinx(audio2)
            MyText = r.recognize_google(audio2)
            
            MyText = MyText.lower()
 
            lcd.message = MyText
            sleep(5)
            lcd.message = "Listening..."
             
    except sr.RequestError as e:
        lcd.message = "Could not request results; {0}".format(e)
         
    except sr.UnknownValueError:
        lcd.message = "unknown error occurred"
