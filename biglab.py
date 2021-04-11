import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from skimage.feature import canny
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.transform import hough_line, hough_line_peaks
from skimage.measure import label, regionprops

from shapely.geometry import LineString, Polygon
from scipy.ndimage.morphology import binary_closing, binary_opening, binary_fill_holes

no_path = "data/no"
yes_path = "data/yes"


# считывание изображений
def read_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


# фильтр Canny
def get_canny(img):
    a_gray = rgb2gray(img)
    a_blur = gaussian(a_gray, 1)
    a_edges = canny(a_blur, sigma=0.1, low_threshold=0.1)
    return a_edges


# получение прямых с помощью преобразования Хафа
def get_hough_transform(img):
    canny_img = get_canny(img)
    h, theta, d = hough_line(canny_img)
    line_peaks = hough_line_peaks(h, theta, d)
    lines = []

    x0 = 0
    x1 = img.shape[1]

    for h, angle, dist in zip(*line_peaks):
        y0 = dist / np.sin(angle)
        y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)
        lines.append((angle, dist, (x0, y0), (x1, y1)))

    # сортируем по местоположению по высоте внутри изображения
    lines = sorted(lines, key=lambda x: abs(x[1]), reverse=True)
    return lines


# отрисовка преобразования Хафа
def show_hough(image, lines):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax[0].set_title('Image')
    ax[0].set_axis_off()

    ax[1].imshow(get_canny(image), cmap=cm.gray)
    for _, _, p0, p1 in lines:
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), '-b')
    ax[1].set_xlim((0, image.shape[1]))
    ax[1].set_ylim((image.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Hough lines')

    plt.tight_layout()


ALLOWED_ANGLE = 1
ALLOWED_DISTANCE = 400
ALLOWED_TURN = 0.2


# функция определения близости прямых

def find_outlier(elem, array):
    flag = 0
    for a in array:
        if abs(abs(int(a[1])) - abs(int(elem[1]))) > ALLOWED_DISTANCE:
            flag += 1
    # вычитаем единицу из-за того, что elem тоже находится в array
    if flag == len(array) - 1:
        # если линия еще и самая нижняя, считаем ее куском пола
        if elem == array[0]:
            return False
    return True


# определение и удаление резкого пересечения прямых внутри изображения
# на предложенном датасете функция не понадобится

def find_intersection(elem, array, img):
    height, width = img.shape[:2]
    for a in array:
        if abs(abs(int(elem[0])) - abs(int(a[0]))) > ALLOWED_TURN:
            # проверяем, лежит ли пересечение внутри изображения
            line1 = LineString([a[2], a[3]])
            line2 = LineString([elem[2], elem[3]])
            inter = line1.intersection(line2)
            if (0 < inter.x < width) or (0 < inter.y < height):
                if abs(a[0]) > abs(elem[0]):
                    return elem
                else:
                    return a
    return None


# удаление лишних прямых, полученных преобразованием Хафа
def remove_noise(line_peaks, img):
    new_lines = line_peaks.copy()

    # удаляем прямые по углу поворота
    for line in line_peaks:
        if abs(line[0]) <= ALLOWED_ANGLE:
            new_lines.remove(line)

    lines_copy = new_lines.copy()

    # удаляем прямые плинтуса/кафеля
    for line in line_peaks:
        if line in new_lines:
            if not find_outlier(line, lines_copy):
                new_lines.remove(line)

    # убираем оставшиеся пересекающиеся прямые
    for line in line_peaks:
        if line in new_lines:
            inter = find_intersection(line, new_lines, img)
            if inter and inter in new_lines:
                new_lines.remove(inter)

    return new_lines


LILAC_MIN = np.array([60, 50, 41], np.uint8)
LILAC_MAX = np.array([179, 255, 255], np.uint8)


# поиск наибольшего объекта маски бинаризации по оттенкам сиреневого
def get_mask(img):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    color_range = cv.inRange(hsv_img, LILAC_MIN, LILAC_MAX)

    mask = binary_closing(color_range, iterations=3)
    img_segment = binary_fill_holes(mask)
    mask_img = binary_opening(img_segment, iterations=30)

    # возьмем наибольший, а остальные удалим

    int_mask = label(mask_img)
    regions = regionprops(int_mask)
    regions = sorted(regions, key=lambda x: x.area, reverse=True)

    if len(regions) > 1:
        for r in regions[1:]:
            int_mask[r.coords[:, 0], r.coords[:, 1]] = 0

    int_mask[int_mask != 0] = 1

    return int_mask


# получение экстремальных точек и ограничивающего прямоугольника таза
def get_bowl_properties(mask):
    contours, _ = cv.findContours(mask, 1, 2)
    cnt = contours[0]
    # ограничивающий прямоугольник
    rect = cv.minAreaRect(cnt)
    # верхняя и нижняя экстремальные точки
    max_p = tuple(cnt[cnt[:, :, 1].argmin()][0])
    min_p = tuple(cnt[cnt[:, :, 1].argmax()][0])

    return max_p, min_p, rect


ALLOWED_PERCENTAGE = 0.75


# определяем, с какой стороны от линии находится точка
def side_of_line(p1, p2, point):
    x, y = point
    x1, y1 = p1
    x2, y2 = p2
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


# поиск точек пересечения прямой стола с ограничивающим прямоугольником таза
def line_crossing(box, p1, p2):
    line = LineString([p1, p2])
    intersect = Polygon(box).intersection(line)

    if intersect:
        return intersect.coords[:]
    else:
        return None


# поиск площади трапеции, лежащей выше прямой стола
def trapezoid_area(rect, p1, p2):
    # находим координаты пересечения линии с ограничивающим прямоугольником
    box = np.int0(cv.boxPoints(rect))
    box_tuples = [tuple(elem) for elem in box]
    l1, l2 = line_crossing(box_tuples, p1, p2)

    # координаты линии зададим против часовой стрелки
    # поскольку box_tuples тоже против часовой стрелки
    polygon = [l2, l1]

    # находим координаты многоугольника, отсекаемого прямой
    for p in box_tuples:
        if side_of_line(l1, l2, p) > 0:
            polygon.append(p)

    # находим площадь многоугольника
    poly = Polygon(polygon)
    return poly.area


# поиск ответа на поставленный вопрос
def get_answer(table, bowl):
    angle, dist, p1, p2 = table
    max_p, min_p, rect = bowl

    if side_of_line(p1, p2, min_p) >= 0:
        return False
    elif side_of_line(p1, p2, max_p) <= 0:
        return True
    else:
        # на предложенном датасете функция не понадобится
        center, wh, angle = rect
        w, h = wh
        area = w * h
        trapezoid = trapezoid_area(rect, p1, p2)
        if area * ALLOWED_PERCENTAGE <= trapezoid:
            return False
        else:
            return True


def print_answer(answer):
    if answer:
        print("Таз можно поместить под стол")
    else:
        print("Таз нельзя поместить под стол")


# проверка всех изображений в датасете
def check_data(data):
    for d in data:
        peaks = get_hough_transform(d)
        table_lines = remove_noise(peaks, d)
        tbl = table_lines[0]

        bowl_mask = get_mask(d)
        clean_mask = bowl_mask.astype(np.uint8)
        bwl = get_bowl_properties(clean_mask)
        ans = get_answer(tbl, bwl)
        print_answer(ans)


# загрузка данных
no_data = read_images(no_path)  # 9 images
yes_data = read_images(yes_path)  # 7 images
print("Проверка датасета с ответом 'нет':")
check_data(no_data)
print("\nПроверка датасета с ответом 'да':")
check_data(yes_data)
