# -*- coding: utf-8 -*-

import cv2


class ShapeDetector:
    """
    Detecta formas geométricas da uma imagem.
    """
    
    def __init__(self):
        pass
    
    def detect(self, contour):
        """
        Detecta a forma de um contour por aproximação.
        :param contour: contorno que deve ser tratado.
        """
        shape = "unidentified"
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        total_vertices = len(approximation)

        if total_vertices == 3:
            shape = "triangle"

        elif total_vertices == 4:
            (x_vrtc, y_vrtc, width, height) = cv2.boundingRect(approximation)
            proporcao = width / float(height)
            shape = "square" if proporcao >= 0.95 and proporcao <= 1.05 else "rectangle"
        
        elif total_vertices == 5:
            shape = "pentagon"
        
        else:
            shape = "circle"
        
        return shape