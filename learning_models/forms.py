from django import forms
from .models import LearningModel


class LearningModelForm(forms.ModelForm):

    class Meta:
        model = LearningModel
        fields = ('model_file', 'title', 'description')
