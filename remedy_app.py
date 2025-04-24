from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import ScreenManager, Screen
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Set background color
Window.clearcolor = (0.9, 0.96, 1, 1)  # Light blue

# Load dataset and train model
data = pd.read_csv("herbal_remedies2.csv")
data_encoded = pd.get_dummies(data, columns=["Condition", "Age_Group", "Dietary_Preferences"])
X = data_encoded.drop(["Remedy_Name", "Ingredients", "Recipe", "Dosage", "Cautions", "Contraindications"], axis=1)
y = data_encoded["Remedy_Name"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Function to get detailed remedy
def get_recommendations(symptoms, age, previous_conditions):
    input_data = pd.DataFrame([[symptoms, age, previous_conditions]],
                              columns=["Condition", "Age_Group", "Dietary_Preferences"])
    input_data_encoded = pd.get_dummies(input_data)
    input_data_encoded = input_data_encoded.reindex(columns=X.columns, fill_value=0)

    predicted_remedy = model.predict(input_data_encoded)[0]
    remedy_details = data[data["Remedy_Name"] == predicted_remedy].iloc[0]

    return {
        "name": predicted_remedy,
        "ingredients": remedy_details["Ingredients"],
        "recipe": remedy_details["Recipe"],
        "dosage": remedy_details["Dosage"],
        "cautions": remedy_details["Cautions"],
        "contraindications": remedy_details["Contraindications"]
    }

# Welcome screen
class WelcomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        layout.add_widget(Image(source='pictures.jpg', size_hint=(1, 0.4)))
        layout.add_widget(Label(text="Welcome to Herbal Care", font_size='24sp', color=(0, 0.4, 0, 1),
                                halign='center', size_hint_y=None, height=100))
        next_button = Button(text="Get Started", size_hint=(1, 0.2), background_color=(0, 0.6, 0.2, 1),
                             color=(1, 1, 1, 1), font_size='18sp', bold=True)
        next_button.bind(on_press=self.go_to_input)
        layout.add_widget(next_button)
        self.add_widget(layout)

    def go_to_input(self, instance):
        App.get_running_app().root.current = 'input'

# Input screen
class InputScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        layout.add_widget(Image(source='pictures.jpg', size_hint=(1, 0.3)))

        layout.add_widget(Label(text="Enter Your Symptoms", font_size='20sp', color=(0, 0.4, 0, 1)))
        self.symptom_input = TextInput(hint_text="e.g., headache, fever", font_size='18sp',
                                       multiline=False, background_color=(0.9, 0.9, 0.9, 1))
        layout.add_widget(self.symptom_input)

        layout.add_widget(Label(text="Enter Your Age Group", font_size='20sp', color=(0, 0.4, 0, 1)))
        self.age_input = TextInput(hint_text="e.g., Adult, Teen, Child", font_size='18sp',
                                   multiline=False, background_color=(0.9, 0.9, 0.9, 1))
        layout.add_widget(self.age_input)

        layout.add_widget(Label(text="Enter Previous Mental Conditions", font_size='20sp', color=(0, 0.4, 0, 1)))
        self.conditions_input = TextInput(hint_text="e.g., anxiety, depression", font_size='18sp',
                                          multiline=False, background_color=(0.9, 0.9, 0.9, 1))
        layout.add_widget(self.conditions_input)

        self.next_button = Button(text="Get Remedy", size_hint=(1, 0.2), background_color=(0, 0.6, 0.2, 1),
                                  color=(1, 1, 1, 1), font_size='18sp', bold=True)
        self.next_button.bind(on_press=self.go_to_remedy)
        layout.add_widget(self.next_button)

        self.add_widget(layout)

    def go_to_remedy(self, instance):
        app = App.get_running_app()
        symptoms = self.symptom_input.text
        age = self.age_input.text
        previous_conditions = self.conditions_input.text
        if symptoms and age and previous_conditions:
            details = get_recommendations(symptoms, age, previous_conditions)
            remedy_text = (
                f"[b]Recommended Remedy:[/b] {details['name']}\n\n"
                f"[b]Ingredients:[/b] {details['ingredients']}\n\n"
                f"[b]Recipe:[/b] {details['recipe']}\n\n"
                f"[b]Dosage:[/b] {details['dosage']}\n\n"
                f"[b]Cautions:[/b] {details['cautions']}\n\n"
                f"[b]Contraindications:[/b] {details['contraindications']}"
            )
            app.root.get_screen('remedy').recommendation_label.text = remedy_text
        else:
            app.root.get_screen('remedy').recommendation_label.text = "[b]Please enter all fields.[/b]"
        app.root.current = 'remedy'

# Remedy screen
class RemedyScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        scroll = ScrollView(size_hint=(1, 0.8))
        self.recommendation_label = Label(text="Your remedy will appear here.", font_size='16sp',
                                          color=(0.2, 0.2, 0.2, 1), markup=True, size_hint_y=None)
        self.recommendation_label.bind(texture_size=self._update_label_height)
        scroll.add_widget(self.recommendation_label)
        layout.add_widget(scroll)

        back_button = Button(text="Back to Input", size_hint=(1, 0.2), background_color=(0.6, 0, 0, 1),
                             color=(1, 1, 1, 1), font_size='18sp', bold=True)
        back_button.bind(on_press=self.go_back_to_input)
        layout.add_widget(back_button)
        self.add_widget(layout)

    def _update_label_height(self, instance, value):
        instance.height = value[1]

    def go_back_to_input(self, instance):
        App.get_running_app().root.current = 'input'

# App setup
class HerbalRemedyApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(WelcomeScreen(name='welcome'))
        sm.add_widget(InputScreen(name='input'))
        sm.add_widget(RemedyScreen(name='remedy'))
        return sm

if __name__ == '__main__':
    HerbalRemedyApp().run()
