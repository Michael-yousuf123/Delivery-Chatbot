from kivymd.app import MDApp
from kivy.lang import Builder 
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager
from kivymd.uix.label import MDLabel
from kivy.properties import StringProperty,NumericProperty
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
        screen_manager.add_widget(Builder.load_file('splash.kv'))
        screen_manager.add_widget(Builder.load_file('bot.kv'))
        return screen_manager
    def on_start(self):
        Clock.schedule_once(self.bot, 5)

    def bot(self, *args): 
        screen_manager.current = 'bot'
if __name__ == '__main__':
    # LabelBase.register(name= "Roboto", fn_regular="Roboto-Regular")
    ChatApp().run()