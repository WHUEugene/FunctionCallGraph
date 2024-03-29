import ast
import os
from graphviz import Digraph
import colorsys
import argparse

class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = {}
        self.current_function_name = None
        self.main_function_name = None
        self.call_order = []  # 用于记录函数调用的顺序
        self.file_of_function = {}  # 记录每个函数所在的文件
        self.main_file = None  # 记录主函数所在的文件
        self.main_functions = []  # 记录所有的主函数
        self.main_files = []  # 记录每个主函数所在的文件

        self.filter = {
    # Python 内置函数
    'abs', 'dict', 'help', 'min', 'setattr', 'all', 'dir', 'hex', 'next', 'slice', 'any',
    'divmod', 'id', 'object', 'sorted', 'ascii', 'enumerate', 'input', 'oct', 'staticmethod',
    'bin', 'eval', 'int', 'open', 'str', 'bool', 'exec', 'isinstance', 'ord', 'sum', 'bytearray',
    'filter', 'issubclass', 'pow', 'super', 'bytes', 'float', 'iter', 'print', 'tuple',
    'callable', 'format', 'len', 'property', 'type', 'chr', 'frozenset', 'list', 'range',
    'vars', 'classmethod', 'getattr', 'locals', 'repr', 'zip', 'compile', 'globals', 'map',
    'reversed', 'import', 'complex', 'hasattr', 'max', 'round', '__import__', 'delattr', 'hash',
    'memoryview', 'set',

    # 常用标准库函数
    'os.path.join', 'os.path.exists', 'os.path.isfile', 'os.path.isdir', 'os.makedirs',
    'sys.exit', 'json.dumps', 'json.loads',

    # 其他常用库函数
    'numpy.array', 'numpy.zeros', 'numpy.ones', 'numpy.empty', 'numpy.dot', 'numpy.linalg.inv',
    'numpy.linalg.eig', 'pandas.DataFrame', 'pandas.read_csv', 'pandas.read_excel',
    'matplotlib.pyplot.plot', 'matplotlib.pyplot.show', 'tqdm.tqdm', 'cv2.imread', 'cv2.imshow',
    
    'tqdm'
}

    def visit_FunctionDef(self, node):
        self.current_function_name = node.name
        self.file_of_function[node.name] = self.current_file  # 记录函数所在的文件
        self.generic_visit(node)
        if self.current_function_name == self.main_function_name:
            self.main_file = self.current_file  # 记录主函数所在的文件

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            called_function_name = node.func.id
            if self.current_function_name and called_function_name not in self.filter:  # 检查被调用的函数名是否在过滤器中
                if self.current_function_name in self.calls:
                    self.calls[self.current_function_name].add(called_function_name)
                else:
                    self.calls[self.current_function_name] = {called_function_name}
                self.call_order.append((self.current_function_name, called_function_name))  # 记录函数调用的顺序
        self.generic_visit(node)

    def visit_If(self, node):
        if isinstance(node.test, ast.Compare):
            if isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__':
                if len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq):
                    if len(node.test.comparators) == 1 and isinstance(node.test.comparators[0], ast.Str) and node.test.comparators[0].s == '__main__':
                        self.main_functions.append(self.current_function_name)  # 记录主函数
                        self.main_files.append(self.current_file)  # 记录主函数所在的文件
        self.generic_visit(node)

def build_call_graph(directory):
    visitor = FunctionCallVisitor()
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(subdir, file)
                visitor.current_file = os.path.relpath(file_path, directory)  # 设置当前文件
                with open(file_path, "r") as source:
                    tree = ast.parse(source.read())
                    visitor.visit(tree)
    return visitor.calls, visitor.main_functions, visitor.main_files, visitor.call_order, visitor.file_of_function  # 返回所有的主函数和他们所在的文件
def get_color(value, max_value):
    if max_value == 0:
        hue = 0
    else:
        hue = (1 - value / max_value) * 0.5  # 将色调限制在0.7以内,避免太亮的颜色
    rgb = colorsys.hsv_to_rgb(hue, 1, 1)
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def draw_graph(calls, main_function_name, call_order, file_of_function, output_file, main_file):  # 接收call_order
    dot = Digraph(comment='Function Call Graph', format='png')
    node_colors = {}
    visited = set()  # 跟踪已访问的节点
    md_lines = []  # 存储Markdown文本的行
    
    max_call_count = max(len(callees) for callees in calls.values())

    def traverse(function_name, parent_function=None, in_progress=None, level=1):
        nonlocal file_of_function  # 使用 nonlocal 关键字来引用外部的 file_of_function 变量
        # file_of_function = file_of_function.get(function_name, '')  # 获取函数所在的文件
        if in_progress is None:
            in_progress = set()

        if function_name in in_progress:  # 如果我们正在处理这个节点，说明我们找到了一个循环
            if parent_function is not None:
                parent_color = node_colors[len(calls[parent_function])]
                dot.edge(function_name, parent_function, color=parent_color, dir='back', constraint='false')
                md_lines.append(f"{'  ' * (level - 1)}-  {parent_function} (循环调用)")  # 在Markdown文本中添加循环调用的标记
            return

        if function_name in visited:  # 如果我们已经处理过这个节点，直接返回
            return

        visited.add(function_name)  # 标记这个节点为已访问
        in_progress.add(function_name)  # 标记这个节点为正在处理

        file_of_current_function = file_of_function.get(function_name, '')  # 获取函数所在的文件
        if file_of_current_function:  # 如果文件路径不为空，添加 '/'
            file_of_current_function += '/'
        
        if function_name not in calls:
            dot.node(function_name, style='filled', fillcolor='#FFFFFF')  # 为不在calls中的函数使用白色
            if parent_function is not None:
                parent_color = node_colors[len(calls[parent_function])]
                dot.edge(parent_function, function_name, color=parent_color)
                md_lines.append(f"{'#' * (level - 1)} {file_of_current_function}{function_name}")  # 在Markdown文本中添加函数调用关系
        else:
            call_count = len(calls[function_name])
            if call_count not in node_colors:
                node_colors[call_count] = get_color(call_count, max_call_count)
            
            color = node_colors[call_count]
            dot.node(function_name, style='filled', fillcolor=color)
            
            if parent_function is not None:
                parent_color = node_colors[len(calls[parent_function])]
                dot.edge(parent_function, function_name, color=parent_color)
                md_lines.append(f"{'#' * (level - 1)} {file_of_current_function}{function_name}")  # 在Markdown文本中添加函数调用关系
            
            # 按照函数在源代码中的位置排序函数调用
            sorted_callees = sorted(calls[function_name], key=lambda x: call_order.index((function_name, x)) if (function_name, x) in call_order else float('inf'))
            for callee in sorted_callees:
                traverse(callee, function_name, in_progress, level + 1)

        in_progress.remove(function_name)  # 标记这个节点为处理完成

    md_lines.append(f"# { main_file}:{main_function_name}")  # 添加主标题，包含主函数所在的文件
    traverse(main_function_name)
    
    return dot, md_lines  # 返回dot对象和Markdown文本行

def main():
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description="Generate a function call graph.")
    
    # 添加directory参数,设置默认值为当前目录
    parser.add_argument('--directory', '-d',
                        type=str,
                        default=os.getcwd(),
                        help='The directory to analyze, defaults to the current working directory.')
    
    # 解析命令行参数
    args = parser.parse_args()
    directory = args.directory
    calls, main_functions, main_files, call_order, file_of_function = build_call_graph(directory)  # 获取所有的主函数和他们所在的文件
    
    # 创建存放结果的子目录
    output_directory = os.path.join(directory, 'FunctionCallGraph')
    os.makedirs(output_directory, exist_ok=True)

    for main_function_name, main_file in zip(main_functions, main_files):  # 为每个主函数生成一个函数调用图
        output_file = f'function_call_graph_{main_function_name}'
        dot, md_lines = draw_graph(calls, main_function_name, call_order, file_of_function, output_file, main_file)
        
        # 使用主函数的名字和文件名作为图的名字
        graph_name = f"{main_file.replace('/', '_')}_{main_function_name}_FunctionCallGraph"
        dot.render(os.path.join(output_directory, graph_name), view=True)  # 保存到指定目录
        
        # 将Markdown文本写入文件
        with open(os.path.join(output_directory, f"{graph_name}.md"), "w") as md_file:  # 保存到指定目录
            md_file.write("\n".join(md_lines))
if __name__ == "__main__":
    main()
