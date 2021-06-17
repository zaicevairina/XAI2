from django.contrib import admin
from home.models import User

class UserAdmin(admin.ModelAdmin):
	  list_display = ('username', )

# Register your models here.

admin.site.register(User, UserAdmin)
