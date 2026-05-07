Source: https://www.youtube.com/watch?v=kCc8FmEb1nY
This is not reasonable to use TypeScript for this task in actual production. The goal is to demonstrate basic logic for developers who know nothing about data science, and who not speak Python.
Can drop your own training data set. I used Tolstoy's War and Peace. A common training data set for this is Tiny Shakespear https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt

This is essentially Andrej Karpathy's NanoGPT (https://www.youtube.com/watch?v=kCc8FmEb1nY), with some tweaks:
* Each slice of each stage of implementation is dynamically introspectable
* More verbose variable names (and code in general)
* Some calculations are implemented with GPU code, in order to make it actually work. But each has its unoptimized TypeScript counterpart
* To make it clear for non-mathematicians, all math operations that are normally taken from libaries (like TensorFlow) are implemented inline (probably you still need a basic idea about matrix multiplication, dot product, and other Algebra 101 stuff)

