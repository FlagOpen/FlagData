## Book_Cleaner
### 描述

`Book_Cleaner` 是一个用于处理电子书的模块，主要功能是从电子书文件（EPUB, MOBI, AZW）中提取文本内容，并且删除一些不需要的部分（目录，广告，图片标注等 ）。

### 环境配置

`Book_Cleaner`使用`Calibre `库 ([[https://calibre-ebook.com/](https://calibre-ebook.com/)]) 对电子书进行解析。

`Calibre `安装：

`sudo -v && wget -nv -O- https://download.calibre-ebook.com/linux-installer.sh | sudo sh /dev/stdin`

### 用法
`Book_Cleaner` 的初始化需要以下三个参数：

1. `config_path`: JSON文件路径，该文件包含了一系列的关键词信息。这些关键词用于指导文本的清洗过程。用户可以编辑这个文件，添加自定义的关键词以适应特定的清洗需求。

2. `workspace`: 一个文件夹路径，用于存储处理电子书过程中生成的临时文件。

   (**注意**: `Book_Cleaner`初始化时会在`workspace`中建立一个名为`book_temp_dir1`的文件夹，每次调用`clean`函数时会删除`book_temp_dir1`里面的所有文件, 若初始化之前已经有名为`book_temp_dir1`的文件夹，则会尝试创建`book_temp_dir2`, 依此类推。)

3. `target_path`: JSON文件路径，处理后内容将被保存在`target_path`中。

通过调用 `Book_Cleaner`的`clean`函数可以实现对电子书的处理，以下是一个简单的使用示例：

```python
# 配置文件路径
config_path = 'path/to/config.json'
# 临时文件路径
workspace = 'path/to/workspace'
# 目标路径
target_path = 'path/to/target'

# 初始化
cleaner = Book_Cleaner(config_path, workspace, target_path)

# 执行清洗处理
cleaner.process_books()
```

### 效果展示
#### 1. 去除封皮，版权页
***
>~~*First published in 2015*~~
>
>~~*Allen & Unwin*~~
>
>~~*83 Alexander Street*~~
>
>~~*Crows Nest NSW 2065*~~
>
>~~*Australia*~~
>
>~~*Phone: (61 2) 8425 0100*~~
>
>~~*Cataloguing-in-Publication details are available from the National Library of Australia*~~
>
>~~*ISBN 978 1 74331 9208*~~
>
>~~*eISBN 978 1 74343 637 0*~~
>
>~~*Internal design by Christabella Designs*~~
>
>~~*Typeset by Post Pre-press Group, Australia*~~
>
>~~*Extracts from The Kite Runner by Khaled Hosseini © Bloomsbury Publishing UK*~~
>
> …
>
>*Chapter 1*
>
>*She was standing at the end of the hall in the late afternoon light, her back arched against the wall, between two landscape paintings. Closing her eyes, she ran her hands up and down the red dress clinging to her curves. The dress had been designed by a genius. Its neckline was low enough to be provocative, but the hemline was below knee-length—too long for Leo, the husband of our hostess, to …*
***
#### 2. 去除目录
***
>~~*Contents*~~
>
>~~*Cover*~~
>
>~~*About the Book*~~
>
>~~*About the Author*~~
>
>~~*Title Page*~~
>
>~~*Dedication*~~
>
>~~*Prologue*~~
>
>~~*Chapter One Innocent Days*~~
>
>~~*Chapter Two Culture Shock*~~
>
>~~*Chapter Three Daddy*~~
>
>…
>
>*About the Book*
>
>*What do they find attractive about me? An underage girl who just lies there sobbing, looking up at them... as they come to me one by one.*
>*This is the shocking true story of how a young girl from Rochdale came to be Girl A – the key witness in the trial of Britain’s most notorious child sex ring. …*
***
#### 3. 去除图片注释
***
>*No disease has ever been so instantly recognized or so widely known and feared. …*
>
>~~*Figure 1. Smallpox Deities. Sopona (left) was the smallpox god among the Yorubas of western Africa. Sitala Mata (right), the Hindu goddess of smallpox, shown astride a donkey, was widely worshipped in temples throughout the Indian countryside.*~~
>
>*Dr. Nick Ward, one of my senior staff, accompanied me on the ward rounds. A veteran of medical service in Africa, he had cared for patients with the worst of tropical diseases. As we left the hospital, he placed his hands on the railing of a balcony, leaned over as he looked at the ground and said, “I don’t think I can ever again walk through a ward like that. It is unimaginable.”*
***

#### 4. 去除广告
***
>~~*Medieval Series*~~
>
>~~*Legendary Bastards of the Crown Series*~~
>
>~~*Legendary Bastards of the Crown Series*~~
>
>~~*Seasons of Fortitude Series*~~
>
>~~*Legacy of the Blade Series*~~
>
>*…*
>
>~~*And More!*~~
>
>~~*Please visit http:// xxx.com !*~~
>
>*…*
>
>*A small hum of excitement rose inside her. She’d been traveling since the morning, and this trip was twice the distance of the longest journey she’d ever taken alone. But she’d done it. Exhausted but quietly exhilarated, she turned the radio up loud. This was a time for celebration. She clenched her hand in the air in a solitary fist-bump.*
>*A wooden sign on the left-hand side of the road read: BLEATH Population 3,667*
>
>~~*Good book recommendation: http:// xxx.com !*~~
>
>*A small town was going to be a culture shock after New York City—the only place she’d ever known. She’d lived in a three-bedroom apartment with her parents and brother since she was three. And now she was heading off by herself for an entire month, to conduct research for her final year thesis at college.*
***


