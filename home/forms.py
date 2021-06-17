from django import forms
from home.models import User

class UserForm(forms.ModelForm):

	class Meta:
		model = User
		fields = ('username', 'password')

class RegistrForm(forms.ModelForm):

	class Meta:
		model = User
		fields = ('username', 'password', 'first_name', 'last_name')