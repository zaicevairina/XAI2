from django.db import models
from home.models import User
from jsonfield import JSONField


def user_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT/user_<id>/<filename>
    return 'user_{0}/{1}'.format(instance.user.id, filename)


class LearningModel(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='attachments',
        verbose_name="пользователь"
    )
    title = models.CharField('название', max_length=128, unique=True)
    model_file = models.FileField(null=True, upload_to=user_directory_path)
    path = models.CharField('путь', max_length=256)
    name = models.CharField('имя', max_length=50, default='heart.csv')
    is_learned = models.BooleanField(default=False)
    description = models.TextField('описание', default='')
    fields = JSONField()

    class Meta:
        verbose_name = "Обученная модель"
        verbose_name_plural = "Обученные модели"
        ordering = ["user"]
