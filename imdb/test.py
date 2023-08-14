import sys
from collections import deque

input = sys.stdin.readline

waiting = deque(map(int, input().split()))
Q = deque()
