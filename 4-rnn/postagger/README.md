# Result

|model|windows|epochs|accuracy|thread|speed|GPU Memory|
|:------|------|------|------|------|------|------|
|mlp|1|100|95.0%|32|0.46h |411M|
|uni-lstm|0|100|93.5%|32|2.3h| 219M |
|uni-lstm|1|100|94.9%|32|2.45h| 219M |
|bi-lstm|0|100|  |32|  |  |
|bi-lstm|1|100|  |32|  |  |
|stacked-lstm|0|100|93.9%|32|4.05h|283M |
|stacked-lstm|1|100|94.5%|32|4.15h|283M |
