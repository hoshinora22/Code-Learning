# CPP笔记

## STL相关

##### std::move

`std::move` 是 C++11 中引入的一个函数模板，它可以将给定对象的值移动到另一个对象中，而不是进行拷贝操作。该函数模板通常用于实现移动语义，可以在某些情况下提高程序的性能。

`std::move` 接受一个参数，这个参数是一个左值引用。当传入的参数是一个左值时，std::move 会将该左值转化为一个右值引用，并返回该右值引用。这个过程并没有实际的拷贝操作，只是改变了左值的类型，因此可以看作是一个移动操作。

使用 `std::move` 可以将一个对象的所有权从一个对象转移到另一个对象，通常在以下情况下使用：

1. **移动语义**：当需要对一个对象进行移动操作时，可以使用 `std::move` 来避免进行不必要的拷贝操作，提高程序的性能。
2. **容器的移动操作**：当需要对容器进行移动操作时，可以使用 `std::move` 来避免进行不必要的拷贝操作，提高程序的性能。

需要注意的是，使用 `std::move` 不会对对象进行任何复制操作，也不会对对象的值进行修改。它只是将对象的值转移到另一个对象中，因此在移动之后原来的对象可能会处于未定义的状态，不能再使用它。

在实际编程中，可以使用 `std::move` 来实现类的移动构造函数和移动赋值运算符，以提高程序的性能。

------

移动语义和容器的移动操作是 C++11 中引入的两个重要特性，它们可以帮助程序员更高效地管理内存和对象的所有权。

**移动语义**指的是将对象的所有权从一个对象转移给另一个对象的操作。在 C++11 之前，所有的对象都是通过拷贝操作进行赋值和传递的，因此在需要频繁地传递和赋值对象时，会出现大量的对象拷贝，从而导致程序的性能下降。移动语义的出现解决了这个问题，通过移动对象而不是拷贝对象来实现更高效的赋值和传递操作。

具体来说，移动语义是通过右值引用来实现的。右值引用是一种特殊的引用类型，可以绑定到一个右值（临时对象或者表达式结果），表示该对象即将被销毁或不再使用，因此可以将其所有权转移给另一个对象。使用 `std::move` 可以将一个对象的所有权从一个对象转移到另一个对象，而不进行任何拷贝操作。

**容器的移动操作**是指将容器中的元素进行移动操作，而不是拷贝操作。在 C++11 中，引入了移动构造函数和移动赋值运算符来支持容器的移动操作。当使用 `std::move` 将一个对象移动到另一个对象时，容器会自动调用相应的移动构造函数和移动赋值运算符来将元素移动到另一个容器中，从而避免了不必要的拷贝操作，提高了程序的性能。

容器的移动操作在以下情况下特别有用：

1. 对于大型对象，移动操作比拷贝操作更高效，可以减少内存分配和释放的次数。
2. 在容器元素类型不支持拷贝操作的情况下，只能使用移动操作进行赋值和传递。

总之，移动语义和容器的移动操作是 C++11 中非常重要的特性，它们可以帮助程序员更高效地管理内存和对象的所有权，提高程序的性能。



当我们需要频繁传递和赋值一个对象时，使用移动语义可以提高程序的性能，下面是一个例子：

```cpp
class MyString {
public:
    MyString() : data_(nullptr), size_(0) {}
    MyString(const char* str) {
        size_ = strlen(str);
        data_ = new char[size_];
        memcpy(data_, str, size_);
    }
    ~MyString() {
        delete[] data_;
    }
    // 移动构造函数
    MyString(MyString&& other) {
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    // 移动赋值运算符
    MyString& operator=(MyString&& other) {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
private:
    char* data_;
    size_t size_;
};

int main() {
    MyString s1("hello");
    MyString s2(std::move(s1));  // 移动构造函数
    MyString s3;
    s3 = std::move(s2);  // 移动赋值运算符
    return 0;
}
```

在上面的代码中，我们定义了一个类 `MyString`，它实现了移动构造函数和移动赋值运算符。在 `main` 函数中，我们创建了三个 `MyString` 对象`s1`、`s2` 和 `s3`，通过使用`std::move` 将 `s1` 和 `s2` 移动到 `s2` 和 `s3` 中，避免了不必要的拷贝操作，提高了程序的性能。

容器的移动操作也非常有用，下面是一个例子：

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<std::string> vec;
    vec.push_back("hello");  // 拷贝构造函数
    vec.push_back("world");  // 拷贝构造函数
    std::vector<std::string> vec2 = std::move(vec);  // 移动构造函数
    std::cout << vec.size() << std::endl;  // 输出 0
    std::cout << vec2.size() << std::endl;  // 输出 2
    return 0;
}
```

在上面的代码中，我们创建了一个 `vector` 容器 `vec`，并向其中插入两个字符串。当我们创建 `vec2` 时，使用 `std::move` 将 `vec` 移动到 `vec2` 中，避免了拷贝操作。在移动之后，`vec` 的大小变为 0，而 `vec2` 的大小为 2。这个例子展示了如何使用容器的移动操作来避免不必要的拷贝操作，提高程序的性能。



## 杂项知识

##### 右值引用

右值引用（Rvalue reference）是 C++11 中新增的一种引用类型，用来**表示一个可以被移动（move）但不能被复制（copy）的对象**。它是通过在类型名前面添加 && 来定义的。

右值引用有以下两个主要用途：

1. 实现**移动语义**：右值引用允许我们将一个对象的状态从一个对象移动到另一个对象，而不是拷贝。在移动对象时，可以通过 `std::move` 函数将一个左值转换为右值引用。
2. 实现**完美转发**：右值引用还允许我们将参数按照原样转发给另一个函数，这被称为完美转发。完美转发允许我们在不改变传递参数的类型和值的情况下，将参数传递给另一个函数。

下面是一个简单的例子，展示了如何使用右值引用来实现移动语义：

```cpp
class MyString {
public:
    // 移动构造函数
    MyString(MyString&& other) {
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
private:
    char* data_;
    size_t size_;
};

int main() {
    MyString s1("hello");
    MyString s2(std::move(s1));  // 移动构造函数
    return 0;
}
```

在上面的代码中，我们定义了一个 `MyString` 类，并实现了移动构造函数。在 `main` 函数中，我们创建了两个 `MyString` 对象 `s1` 和 `s2`，通过使用 `std::move` 将 `s1` 移动到 `s2` 中，避免了不必要的拷贝操作。



下面是一个简单的例子，展示了如何使用右值引用来实现完美转发：

```cpp
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

int main() {
    auto p = make_unique<MyString>("hello");  // 调用 make_unique 函数
    return 0;
}
```

在上面的代码中，我们定义了一个 `make_unique` 函数，该函数接受任意数量的参数，并通过 `std::forward` 将这些参数按原样转发给 `std::unique_ptr` 构造函数。这样，我们可以使用该函数来创建任何类型的对象，而不必写出每个类型的特定构造函数。



##### 完美转发

完美转发（Perfect forwarding）是指在函数调用中，将原始的参数类型和值，精确地转发给另一个函数，而不会丢失其类型或值。这种技术通常使用右值引用来实现，它允许我们将一个左值转换为右值引用，并通过 `std::forward` 函数将其转发给另一个函数。

完美转发通常用于以下两个场景：

1. 函数模板：当我们编写一个函数模板时，我们不知道实际参数的类型，但是我们需要精确地将这些参数转发给其他函数。在这种情况下，我们可以使用右值引用和 `std::forward` 来实现完美转发。
2. 转发构造函数和转发赋值运算符：当我们定义一个类，并希望它的构造函数和赋值运算符可以接受任意类型的参数时，我们可以使用完美转发来实现。这样，我们可以将参数精确地转发给其他构造函数或赋值运算符。

下面是一个简单的例子，展示了如何使用完美转发来编写一个函数模板：

```cpp
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

int main() {
    auto p = make_unique<MyString>("hello");  // 调用 make_unique 函数
    return 0;
}



class Path {
    
private:
	std::vector<Point> m_path;
}

template<class... Args>
void emplace_back(Args &&... args) {
    m_path.emplace_back(std::forward<Args>(args)...);
}
```

在上面的代码中，我们定义了一个 `make_unique` 函数模板，它接受任意数量的参数，并通过 `std::forward` 将这些参数按原样转发给 `std::unique_ptr` 构造函数。这样，我们可以使用该函数来创建任何类型的对象，而不必写出每个类型的特定构造函数。

下面是另一个简单的例子，展示了如何使用完美转发来编写一个转发构造函数：

```cpp
class MyString {
public:
    template<typename... Args>
    MyString(Args&&... args) : data_(std::forward<Args>(args)...) {}

private:
    std::string data_;
};

int main() {
    MyString s1("hello");
    MyString s2(s1);  // 调用拷贝构造函数
    MyString s3(std::move(s1));  // 调用移动构造函数
    return 0;
}
```

在上面的代码中，我们定义了一个 `MyString` 类，并使用完美转发来实现其构造函数。该构造函数接受任意数量的参数，并将它们精确地转发给 `std::string` 构造函数。这样，我们可以使用该构造函数来创建任何类型的 `MyString` 对象，而不必写出每个类型的特定构造函数。

-------

在 C++ 中，`Args` 是一个通用的模板类型参数，通常用于函数模板或类模板中。它代表一个可变数量的参数包（Parameter Pack），即一组类型或值的列表。

在函数模板中，`Args` 通常用于表示函数的参数列表。由于函数模板的参数数量和类型是不确定的，因此使用 `Args` 可以接受任意数量和类型的参数，并将它们精确地转发给其他函数。

在类模板中，`Args` 通常用于表示类的模板参数列表。由于类模板的参数数量和类型也是不确定的，因此使用 `Args` 可以定义一个通用的类，可以接受任意数量和类型的模板参数。

`Args` 是一个特殊的模板参数类型，它通常与其他模板参数类型一起使用，例如 `typename T` 和 `typename... Ts`。在函数模板中，`typename... Ts` 通常用于表示可变参数列表，而 `Args` 则用于将这些参数列表传递给其他函数。在类模板中，`typename... Ts` 通常用于表示可变模板参数列表，而 `Args` 则用于将这些模板参数列表传递给其他类。



下面是一些使用 `Args` 的示例：

1. 函数模板中的 `Args`

```cpp
// 接受任意数量的参数，并打印出它们的值
template<typename... Args>
void print_values(Args... args) {
    std::cout << "Values: ";
    (std::cout << ... << args) << std::endl;
}

int main() {
    // 调用 print_values 函数，传递不同类型和数量的参数
    print_values(1, "hello", 3.14, 'a');
    return 0;
}
```

输出：

```makefile
Values: 1hello3.14a
```



2. 类模板中的 `Args`

```cpp
// 定义一个通用的类模板，可以接受任意数量和类型的模板参数
template<typename... Args>
class my_class {
public:
    // 构造函数，接受任意数量和类型的参数
    my_class(Args... args) {
        std::cout << "my_class constructor called with " << sizeof...(Args) << " arguments." << std::endl;
    }
};

int main() {
    // 实例化 my_class 类，传递不同数量和类型的模板参数
    my_class<int, double> c1(1, 3.14);
    my_class<std::string, char, float> c2("hello", 'a', 3.14f);
    return 0;
}
```

输出：

```makefile
my_class constructor called with 2 arguments.
my_class constructor called with 3 arguments.
```
