from kivymd.app import MDApp
from kivy.lang import Builder 
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager
from kivy.clock import Clock
Window.size = (360, 640)
# 40
class ChatApp(MDApp):

    def build(self):
        global screen_manager
        screen_manager = ScreenManager()
        screen_manager.add_widget(Builder.load_file('splash.kv'))
        screen_manager.add_widget(Builder.load_file('bot.kv'))
        return
    def on_start(self):
        Clock.schedule_once(self.bot, 5)

    def bot(self, *args): 
        screen_manager.current = 'bot'
if __name__ == '__main__':
    ChatApp().run()