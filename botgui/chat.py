import sys
sys.path.insert(0, "/home/miki/Desktop/Deployment/delivery-chatbot")
from kivymd.app import MDApp
from kivy.lang import Builder 
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager
from kivymd.uix.label import MDLabel
from kivy.properties import StringProperty,NumericProperty
from api.bot import bot_name, get_response
from kivy.clock import Clock
Window.size = (360, 640)
# 40   
class Command(MDLabel):
    text = StringProperty()
    size_hint_x = NumericProperty()
    halign = StringProperty
    font_name = "Roboto"
    font_size = 17
class Response(MDLabel):
    text = StringProperty()
    size_hint_x = NumericProperty()
    halign = StringProperty
    font_name = "Roboto"
    font_size = 17

class ChatApp(MDApp):

    def build(self):
        global screen_manager
        screen_manager = ScreenManager()
 
        screen_manager.add_widget(Builder.load_file('bot.kv'))
        return screen_manager
    def response(self, msg):
        if screen_manager.get_screen('bot').bot_name.text == bot_name:
            response = get_response(msg)
            screen_manager.get_screen('bot').chat_list.add_widget(Response(text=response, size_hint_x=.75))
    def send(self):
        if screen_manager.get_screen('bot').text_input != "":
            msg = screen_manager.get_screen('bot').text_input.text
            screen_manager.get_screen('bot').chat_list.add_widget(Command(text=msg, size_hint_x=.2, halign='center'))
            Clock.schedule_once(self.response, 2)
            screen_manager.get_screen('bot').text_input = ''
    
    def bot(self, *args): 
        screen_manager.current = 'bot'
if __name__ == '__main__':
    # LabelBase.register(name= "Roboto", fn_regular="Roboto-Regular")
    ChatApp().run()