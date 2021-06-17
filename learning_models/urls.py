from .views import get_all_models, create_model, create_model2, learn_model, get_model_info, check_model, tree, compare_models, get_learn_error
from django.urls import path

urlpatterns = [
    path('', get_all_models, name='get_all'),
    path('create/', create_model, name='create'),
    path('create/fix', create_model2, name='create2'),
    path('<int:model_id>/', check_model, name='check_model'),
    path('info/<int:model_id>', get_model_info, name='info_model'),
    path('learn/<int:model_id>', learn_model, name='learn_model'),
    path('tree/<int:model_id>', tree, name='tree'),
    path('compare', compare_models, name='compare'),
    path('get_learn_error', get_learn_error, name='get_learn_error'),
]
