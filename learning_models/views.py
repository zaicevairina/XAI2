import os
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseNotAllowed, HttpResponse, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from .models import LearningModel
from .forms import LearningModelForm
from django.conf import settings
import pandas as pd
import csv
from copy import deepcopy
import json
from django.forms.models import model_to_dict
from  xai import fit_model, predict, XAI,predict_some_trees, show_tree, info, interaction, shap_summary

ALLOWED_EXTENSIONS = set(['csv'])


@login_required
def get_all_models(request):
    if request.method == 'GET':
        user = request.user
        learning_models = LearningModel.objects.filter(user=user).values("id", 'title')
        return render(request, 'learning_models/get_all.html', {'models': learning_models})

@login_required
def compare_models(request):
    if request.method == 'GET':
        user = request.user
        learning_models = LearningModel.objects.filter(user=user).values("id", 'title', 'name')
        return render(request, 'learning_models/compare.html', {'models': learning_models})

@login_required
def get_learn_error(request):
    if request.method == 'GET':
        name = request.GET.get('name', '')

        graph_data = []

        try:
            with open(f'{name}/learn_error.tsv', 'r', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter='\t')
                for row in reader:
                    graph_data.append(row)

                graph_data = graph_data[1:]

                graph_data_int = []

                for item in graph_data:
                    graph_data_int.append((int(item[0]), float(item[1])))
        except:
            return JsonResponse({ 'error': 'error' })

        return JsonResponse({'graphData': graph_data_int})

@login_required
# @xframe_options_exempt
@csrf_exempt
def check_model(request, model_id):
    if request.method == 'GET':
        user = request.user
        learning_models = LearningModel.objects.filter(user=user).filter(id=model_id).values('id', 'title', 'fields')
        if len(learning_models) == 0:
            return JsonResponse({'nothing': 'nothing'})
        fields = learning_models[0]['fields']
        return render(request, 'learning_models/check_model.html', {'fields': fields})
    elif request.method == 'POST':
        request_fields = request.POST
        user = request.user
        learning_models = LearningModel.objects.filter(user=user).filter(id=model_id)
        if len(learning_models) == 0:
            return JsonResponse({'nothing': 'nothing'})

        for learning_model in learning_models:
            fields = learning_model.fields

            path = learning_model.path
            name = learning_model.name
        value_fields = []
        for field in fields:
            value = request_fields[field]

            if value.isnumeric():
                value = int(value)
            elif value.replace('.', '').replace('-', '').isnumeric():
                value = float(value)
            value_fields.append(value)

        res = predict(name, path)

        probability = res[0]
        class1 = res[1]
        raw = res[2]
        iframe_path = XAI(name, path)


        return render(request, 'learning_models/result.html', {
            'probability': probability,
            'class': class1,
            'raw': raw,
            'path': iframe_path,
        })

@login_required
@csrf_exempt
def create_model(request):
    if request.method == 'GET':
        form = LearningModelForm()
        return render(request, 'learning_models/create_model.html', {'form': form})
    elif request.method == 'POST':

        form = LearningModelForm(request.POST, request.FILES)
        if form.is_valid():
            df = pd.read_csv(form.cleaned_data['model_file'])
            columns = request.POST['columns'].split(',')
            columns.append('target')
            df = df[columns]

            name=request.FILES['model_file'].name
            df.to_csv(str(settings.BASE_DIR) + f'/media/static_csv/{name}', index=False)
            columns = request.POST['columns'].split(',')
            new_model = LearningModel(
                user=request.user,
                path=str(settings.BASE_DIR) + f'/media/static_csv/{name}',
                title=form.cleaned_data['title'],
                name=name,
                description=form.cleaned_data['description'],
                fields = columns,
            )

            new_model.save()

            return JsonResponse({ 'id': new_model.id })
        else:
            return JsonResponse({ 'error': 'error' })

    else:
        HttpResponseNotAllowed(['GET', 'POST'])


@login_required
@csrf_exempt
def create_model2(request):
    if request.method == 'GET':
        form = LearningModelForm()
        return render(request, 'learning_models/create_model.html')
    elif request.method == 'POST':

        form = LearningModelForm(request.POST, request.FILES)
        if form.is_valid():
            df = pd.read_csv(form.cleaned_data['model_file'])
            del df['target']
            names = list(df.columns)
            return JsonResponse({ 'columns': names })
        else:
            return JsonResponse({ 'error': 'error' })
    else:
        HttpResponseNotAllowed(['GET', 'POST'])


@login_required
@csrf_exempt
def learn_model(request, model_id):
    if request.method == 'GET':
        user = request.user
        learning_models = LearningModel.objects.filter(user=user).filter(id=model_id)
        if len(learning_models) == 0:
            return HttpResponseRedirect({'nothing': 'nothing'})
        for learning_model in learning_models:
            model = learning_model
            path = learning_model.path

        if model.is_learned:
            return HttpResponseRedirect('/models/')

        df = pd.read_csv(path)

        del df['target']

        categorical_columns = [c for c in df.columns if df[c].dtype.name == 'object']
        numerical_columns = [c for c in df.columns if df[c].dtype.name != 'object']
        t = []
        for column in df.columns:
            if column in categorical_columns:
                t.append((column, True))
            elif column in numerical_columns:
                t.append((column, False))

        return render(request, 'learning_models/learn_model.html', {
            'columns': t,
        })
    elif request.method == 'POST':
        user = request.user
        learning_models = LearningModel.objects.filter(user=user).filter(id=model_id)
        if len(learning_models) == 0:
            return JsonResponse({'nothing': 'nothing'})

        data = deepcopy(request.POST)
        param = {}
        param['iterations'] = int(data['iterations'])
        data.pop('iterations')
        param['learning_rate'] = float(data['learning_rate'])
        data.pop('learning_rate')
        param['depth'] = int(data['depth'])
        data.pop('depth')
        param['l2_leaf_reg'] = float(data['l2_leaf_reg'])
        data.pop('l2_leaf_reg')
        param['rsm'] = float(data['rsm'])
        data.pop('rsm')
        param['nan_mode'] = data['nan_mode']
        data.pop('nan_mode')
        param['has_time'] = bool(data['has_time'])
        data.pop('has_time')
        param['one_hot_max_size'] = int(data['one_hot_max_size'])
        data.pop('one_hot_max_size')
        param['bagging_temperature'] = float(data['bagging_temperature'])
        data.pop('bagging_temperature')
        param['min_data_in_leaf'] = int(data['min_data_in_leaf'])
        data.pop('min_data_in_leaf')
        param['max_leaves'] = int(data['max_leaves'])
        data.pop('max_leaves')
        print(data)
        data.pop('csrfmiddlewaretoken')
        # Добавить в параметры
        param['cat_features'] = []

        for key in data:
            param['cat_features'].append(key)

        print(param)

        for learning_model in learning_models:
            model = learning_model
            path = learning_model.path
            name = learning_model.name

        result = fit_model(name, path, param)

        model.is_learned = True
        model.save()

        graph_data = []

        with open(f'{name}/learn_error.tsv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                graph_data.append(row)

            graph_data = graph_data[1:]

            graph_data_int = []

            for item in graph_data:
                graph_data_int.append((int(item[0]), float(item[1])))

        return render(request, 'learning_models/learn_result.html', {
            'msg': result,
            'graph_data': json.dumps(graph_data_int),
        })

    else:
        HttpResponseNotAllowed(['GET', 'POST'])

@login_required
@csrf_exempt
def get_model_info(request, model_id):
    if request.method == 'GET':
        user = request.user
        learning_models = LearningModel.objects.filter(user=user).filter(id=model_id)
        if len(learning_models) == 0:
            return HttpResponseRedirect({'nothing': 'nothing'})
        for learning_model in learning_models:
            model = learning_model

        model_dict = model_to_dict(model)
        graph_data = []
        with open(f'{model.name}/learn_error.tsv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                graph_data.append(row)

            graph_data = graph_data[1:]

            graph_data_int = []

            for item in graph_data:
                graph_data_int.append((int(item[0]), float(item[1])))

        res = info(model.name)

        text_info = res[:-3]
        params = res[-1]
        table = res[-3].values.tolist()
        table2 = res[-2].values.tolist()

        path = interaction(model.name)

        path2 = shap_summary(model.name)

        return render(request, 'learning_models/info_model.html', {
            'model': model_dict,
            'graph_data': json.dumps(graph_data_int),
            'text_info': text_info,
            'params': params,
            'table': table,
            'table2': table2,
            'path': path,
            'path2': path2,
        })

    else:
        HttpResponseNotAllowed(['GET'])

@login_required
def tree(request, model_id):
    if request.method == 'GET':
        user = request.user
        learning_models = LearningModel.objects.filter(user=user).filter(id=model_id)
        if len(learning_models) == 0:
            return HttpResponseRedirect({'nothing': 'nothing'})
        for learning_model in learning_models:
            model = learning_model

        tree_str = request.GET.get('tree', '')

        if tree_str == '':
            return JsonResponse({'error': 'error'})

        tree = int(tree_str)

        res = predict_some_trees(model.name, tree, tree + 1, '')
        path = show_tree(model.name, tree)

        print(res)


        probability = list(res[0])
        class1 = str(res[1])
        raw = str(res[2])

        return JsonResponse({
            'probability': probability,
            'class': class1,
            'raw': raw,
            'path': path,
        })

    else:
        HttpResponseNotAllowed(['GET'])
