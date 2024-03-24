# Memory Model in C++

> Where are the variables stored in the memory?

## overall concept

![image-20240323002316416](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20240323002316416.png)

### 关于 Global Vars

- vars defined outside any functions
- can by shared btw .cpp files
- extern声明

![image-20240323002832422](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20240323002832422.png)

### 关于 Static Vars

#### static global variable

- static global variable inhibits access from outside the .cpp file 
- so as the static function

#### static local variable

- static local variable keeps value in between visits to the same function
- is initialized at its first access

![image-20240323004950315](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20240323004950315.png)

> 上面列举三中变量都在code/data区（静态数据区），程序启动时即分配好。

### Pointers to objects

> 之前学过，略

### Defining Reference

- References are a new data type in C++

```c++
char c;			// a character
char* p = &c;	// a pointer to c
char& r = c;	// a reference to c
```

- Declares a new name for an existing object

- Rules of references:

  - References must be initialized when created

  - Initialization establishes a binding

  - Bindings don’t change at run time, unlike pointers

  - Assignment changes the object referred-to

  - The target of a non-const reference must be an **lvalue**.

    ![image-20240323012440987](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20240323012440987.png)

  - ![image-20240323012833494](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20240323012833494.png)

![image-20240323012641752](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20240323012641752.png)

### Dynamic memory allocation

- new expression

  ```c++
  new int;
  new Stash;
  new int[10];
  ```

- delete expression

  ```c++
  delete p;
  delete [] p;
  ```

> 为什么在已经有 malloc / free 的基础上还要 new / delete?

**Answer:  `new \ delete` 在分配\释放内存的同时，还会对所声明的数据类型进行初始化 （调用它的Constructor和Destructor）**

### Const







