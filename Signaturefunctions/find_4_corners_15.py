import numpy as np


def compute_upper_point(input_gray):
    try:
        for i in range(15,input_gray.shape[0]-15,1):
            for j in range(15,input_gray.shape[1]-15,1):
                if (input_gray[i, j] == 100):
                    return i
    except Getoutofloop:
        pass


def compute_down_point(input_gray):
    try:
        for i in range(input_gray.shape[0]-16,15,-1):
            for j in range(input_gray.shape[1]-31,15,-1):
                if (input_gray[i, j] == 100):
                    return i
    except Getoutofloop:
        pass

def compute_right_point(input_gray):
    try:
        for j in range(input_gray.shape[1]-16,15,-1):
            for i in range(input_gray.shape[0]-31,15,-1):
                if (input_gray[i, j] == 100):
                    return j
    except Getoutofloop:
        pass

def compute_left_point(input_gray):
    try:
        for j in range(15,input_gray.shape[1]-15,1):
            for i in range(15,input_gray.shape[0]-15,1):
                if (input_gray[i, j] == 100):
                    return j
    except Getoutofloop:
        pass


def avg_line(input_line,m,c):
    sum = 0
    for i in input_line:
        output = m * i + c
        sum = sum + output
    avg_l = round(sum / len(input_line))
    return avg_l


class Getoutofloop(Exception):
    pass


def points(gray):
    point = []
    down_point = compute_down_point(gray)
    left_point = compute_left_point(gray)
    upper_point = compute_upper_point(gray)
    right_point = compute_right_point(gray)
    point.append(upper_point)
    point.append(down_point)
    point.append(left_point)
    point.append(right_point)
    return point