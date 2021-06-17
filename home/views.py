
from django.http import JsonResponse, HttpResponseNotAllowed, HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .forms import UserForm, RegistrForm
from .models import User
from django.contrib.auth import authenticate, login
from django.views.decorators.csrf import csrf_exempt
import json

# Create your views here.
@login_required
def home(request):
    return render(request, 'home/home.html')

@csrf_exempt
def login1(request):
    if request.method == 'GET':
        form = UserForm()
        return render(request, 'home/login.html', {'form': form})
    elif request.method == 'POST':
        user = authenticate(request, username=request.POST['username'], password=request.POST['password'])
        if user is not None:
            login(request, user)
            return HttpResponseRedirect('/')
        return render(request, 'home/login.html', {'error': "Ошибка авторизации"})
    else:
        HttpResponseNotAllowed(['GET', 'POST'])

@csrf_exempt
def registr(request):
    if request.method == 'GET':
        form = RegistrForm()
        return render(request, 'home/regis.html', {'form': form})
    elif request.method == 'POST':
        form = RegistrForm(request.POST)
        if form.is_valid() and (form.cleaned_data['password'] == request.POST['password2']):
            user = User(username=form.cleaned_data['username'], first_name=form.cleaned_data['first_name'], last_name=form.cleaned_data['last_name'])
            user.set_password(form.cleaned_data['password'])
            user.save()
            login(request, user)
        else:
            print(form.errors.as_data())
            return render(request, 'home/regis.html', {'form': form})
        return HttpResponseRedirect('/')

    else:
        HttpResponseNotAllowed(['GET', 'POST'])

@login_required
def returnJson(request):
    if request.method == 'GET':
        users = User.objects.all().values("first_name", "last_name", "id")
        users = (list(users))

        return JsonResponse({'users': users})
