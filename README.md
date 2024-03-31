def draw_dashed_rectangle(image, pt1, pt2, color, thickness=1, dash_length=10, gap_length=5):
    x1, y1 = pt1
    x2, y2 = pt2
    line_type = cv2.LINE_AA  # 使用抗锯齿线型

    # 计算矩形边框的长度和高度
    width = x2 - x1
    height = y2 - y1

    # 绘制上边框
    for i in range(x1, x2, dash_length + gap_length):
        dash_end = min(i + dash_length, x2)
        cv2.line(image, (i, y1), (dash_end, y1), color, thickness, line_type)

    # 绘制下边框
    for i in range(x1, x2, dash_length + gap_length):
        dash_end = min(i + dash_length, x2)
        cv2.line(image, (i, y2), (dash_end, y2), color, thickness, line_type)

    # 绘制左边框
    for i in range(y1, y2, dash_length + gap_length):
        dash_end = min(i + dash_length, y2)
        cv2.line(image, (x1, i), (x1, dash_end), color, thickness, line_type)

    # 绘制右边框
    for i in range(y1, y2, dash_length + gap_length):
        dash_end = min(i + dash_length, y2)
        cv2.line(image, (x2, i), (x2, dash_end), color, thickness, line_type)
