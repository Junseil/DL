import sys

input = sys.stdin.readline

stack = []

n = int(input())
for _ in range(n):
    x = input().rstrip()
    if len(x) > 2:
        stack.append(int(x[2:]))
    else:
        x = int(x)
        if x == 2:
            if stack:
                print(stack.pop())
            else:
                print(-1)
        if x == 3:
            print(len(stack))
        if x == 4:
            if stack:
                print(0)
            else:
                print(1)
        if x == 5:
            if stack:
                print(stack[-1])
            else:
                print(-1)