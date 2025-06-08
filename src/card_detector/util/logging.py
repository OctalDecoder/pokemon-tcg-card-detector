def print_section_header(title, width=100, sep='-'):
    side_len = (width - len(title) - 2) // 2
    line = f"\n{sep * side_len} {title} {sep * side_len}\n"
    print(line)