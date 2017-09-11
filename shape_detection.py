#-*- coding: utf-8 -*-

import argparse
import os

import cv2
import imutils

from pyimagesearch.shapedetector import ShapeDetector


def buy_path_and_filename_to_dump(filename, dirname=('dump',), root='./'):
    """
    Constroi o path para salvar os arquivos no path especificado.

    :param filename: string com nome do arquivo.
    :param dirname: tupla com nome dos diretórios.
    :param root: diretório base.
    :return: path completo com nome do arquivo no final.
    """
    path = os.path.join(root, 'dump', filename)
    return path


def get_processed_image(image_path):
    """
    Altera a imagem para escala de cor cinza, aplique um pouco de blur e remova os ruídos.

    :param image_path: caminho para a imagem a ser tratada.
    :return: image processada com threshold aplicado.
    """
    gray_scaled_image = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(buy_path_and_filename_to_dump('gray_scaled_image.png'), gray_scaled_image)

    blurred_image = cv2.GaussianBlur(gray_scaled_image, (5, 5), 0)
    cv2.imwrite(buy_path_and_filename_to_dump('blurred_image.png'), blurred_image)

    thresholded_image = cv2.threshold(blurred_image, 60, 255, cv2.THRESH_BINARY)[1]

    return thresholded_image


def get_contours_of_thresholded_image(image):
    """
    Identifica e extrai os contornos da imagem.
    :param image: imagem com threshold (remoção de ruídos) aplicada.
    :return: lista de contornos encontrados na imagem.
    """
    conts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = conts[0] if imutils.is_cv2() else conts[1]

    return conts


if __name__ == '__main__':
    # Constroi um parse de argumentos.
    parse = argparse.ArgumentParser(description="Find centroid of shapes on images")
    parse.add_argument("-i", "--image", required=True, help="path to the input image")
    arguments = vars(parse.parse_args())

    # Carrega a imagem e salva uma cópia do original
    image = cv2.imread(arguments["image"])
    resized_image = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized_image.shape[0])
    cv2.imwrite(buy_path_and_filename_to_dump('original_image.png'), image)

    processed_image = get_processed_image(resized_image)
    contours = get_contours_of_thresholded_image(processed_image)

    shape_detector = ShapeDetector()

    for contour in contours:
        # Calcula o centro do contour
        moment = cv2.moments(contour)
        if moment['m00'] != 0:
            contour_x = int((moment['m10'] / moment['m00']) * ratio)
            contour_y = int((moment['m01'] / moment['m00']) * ratio)

        else:
            contour_x, contour_y = 0, 0

        # Detecta o shape
        shape = shape_detector.detect(contour)

        # Prepara o tamanho da contour, Desenha o contour no centro da forma na imagem original
        contour = contour.astype('float')
        contour *= ratio
        contour = contour.astype('int')

        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        # cv2.circle(image, (contour_x, contour_y), 7, (255, 255, 255), -1)
        cv2.putText(image, shape, (contour_x, contour_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Mostra a imagem
        cv2.imshow("Image", image)
        cv2.waitKey(0)
