# PTB
1. penn.train.pos
2. penn.devel.pos
3. penn.test.pos.blind

# statistic data

1. label_hash.pkl:  {label:index}
2. word_count.pkl:  {word:total_nums}
3. word_hash.pkl:   {word:hash}

all data structure of pkl files is dict(include Counter).
All words whose total_nums is 1 are token as **UNK**
