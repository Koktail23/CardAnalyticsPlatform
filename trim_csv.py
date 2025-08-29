# Самый быстрый способ для больших файлов
with open('synth_data_10m.csv', 'r', encoding='utf-8') as infile:
    with open('output.csv', 'w', encoding='utf-8') as outfile:
        for i in range(100001):  # заголовок + 20 строк
            line = infile.readline()
            if not line:
                break
            outfile.write(line)

print("Готово! Файл output.csv создан с первыми 20 строками")