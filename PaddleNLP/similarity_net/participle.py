import paddlehub as hub

# 首先准备好我们要进行分词的素材
raw_data = [
    ["你觉得明天是个晴天吗","我看还是下雨的可能性大"],
    ["中国哪家公司的人工智能最牛呢"],
    ["我在山上看见爱因斯坦"],
    ["我把车把一把把住了"]
]

# 然后直接调用 PaddleHub 中现成的分词模型 LAC
lac = hub.Module(name="lac")

for texts in raw_data: # 每一次取一个列表中的 元素，这个 元素 是个 字符串 的 列表
    results = lac.lexical_analysis(texts=texts, use_gpu=False, batch_size=1)
    # lexical_analysis(texts=[], data={}, use_gpu=False, batch_size=1, user_dict=None, return_tag=True)
    # lac预测接口，预测输入句子的分词结果
    # texts(list): 待预测数据，如果使用texts参数，则不用传入data参数，二选一即可
    # data(dict): 预测数据，key必须为text，value是带预测数据。如果使用data参数，则不用传入texts参数，二选一即可。
    # 建议使用texts参数，data参数后续会废弃。
    # use_gpu(bool): 是否使用GPU预测
    # batch_size(int): 批处理大小
    # user_dict(None): 该参数不推荐使用，请在使用lexical_analysis()方法之前调用set_user_dict()方法设置自定义词典
    # return_tag(bool): 预测结果是否需要返回分词标签结果
    # 返回结果：results(list): 分词结果是个列表

    for result in results: # 取得结果列表中的一个元素
        print(result)
        # 这里 单个分词 的结果是个字典，其中两个key，一个是分词结果 "word"，一个是词性标注 "tag"
