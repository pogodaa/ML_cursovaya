import os
print("Текущая директория:", os.getcwd())
print("Файлы в текущей директории:", os.listdir('.'))
print("Файлы на уровень выше:", os.listdir('..'))