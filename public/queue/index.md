# Simple and efficient purely functional queues and deques


纯函数式实现队列（queue）和双端队列（deque），使得单次操作的最坏复杂度为 $O(1)$。

<!--more-->

## 路径 

- 经典方法 paired list：单次操作的均摊复杂度 $O(1)$，但最坏复杂度为 $O(n)$
- 使用lazy list替换list：单次操作的最坏复杂度为 $O(\log{n})$
- 使用incrementally pre-evaluating lazy list：单次操作的最坏复杂度为 $O(1)$
- 从queue拓展到deque

## 实现

### Paired List

标准的paired list的实现方式是使用两个list（或者说stack）$\langle L,R\rangle$ 分别存储当前队列 $q$ 的左右子区间，其中 $L$ 是倒序的（栈顶为`q.front`），$R$ 是正序的（栈顶为`q.back`）

这个trick很经典，直接给出操作函数：
$$
\begin{aligned}
f(x)&=1 \\
&=2 \\
&=3 \\
\end{aligned}
$$

具体解释为：

- 执行 $\text{insert}$ 时，在 $R$ 的栈顶处插入，即 $\langle L,e:R\rangle$

- 执行 $\text{remove}$ 时：
  - 若 $L$ 非空，则将 $L$ 的栈顶元素弹出，即 $\langle \text{tl}\ L,R\rangle$
  - 若 $L$ 为空，则将 $R$ 中元素依次弹出，并插入 $L$ 中（由于stack是FILO的，因此相当于将R翻转后赋值给L，并清空R，即$\langle\text{rev}\ R,[]\rangle$，再得到 $\langle \text{tl}\ L,R\rangle$

由于每个元素至多在 $L,R$ 插入/弹出各一次，因此单次操作的均摊时间复杂度为 $O(1)$。

但是这个做法有一个问题：当执行 $\text{remove}$ 且 $L$ 为空的情况时，单次操作需要 $O(|R|)$ 的时间（即翻转列表/栈中元素素全部弹出），这在部分应用场景下不可接受。

#### 代码实现

```haskell
-- 使用元组来表示成对的列表
type PairedList a = ([a], [a])

-- 队列类型
newtype Queue a = Queue (PairedList a) deriving (Show)

-- 创建一个空队列
emptyQueue :: Queue a
emptyQueue = Queue ([], [])

-- 将元素添加到队列末尾
push :: a -> Queue a -> Queue a
push x (Queue (front, rear)) = Queue (front, x:rear)

front :: Queue a -> Maybe (a, Queue a)
front (Queue ([], [])) = Nothing
front (Queue (ll, rr)) =
  case ll of
    [] -> front (Queue (reverse rr, []))
    xs -> Just (head xs, Queue (xs, rr)) 

-- 从队列前部取出元素，并返回新的队列
pop :: Queue a -> Maybe (Queue a)
pop (Queue ([], [])) = Nothing
pop (Queue (front, rear)) =
  case front of
    [] -> pop (Queue (reverse rear, []))
    (x:xs) -> Just (Queue (xs, rear))

-- 检查队列是否为空
empty :: Queue a -> Bool
empty (Queue ([], [])) = True
empty _ = False

size :: Queue a -> Int
size (Queue (x, y)) = length x + length y
```

### Paired Lazy List(Stream)

Lazy List的优势在于操作可以延迟，直到对应元素访问时才执行，且中间结果是记忆化的。

对于Lazy List：
$$
\begin{align*}
X\texttt{++} Y&=Y&&\{|X|=0\}\\
&=\text{hd}\ X:(\text{tl}\ X\texttt{++}Y)&&\{|X|>0\}\\

\text{rev}\ X &= \text{rev}'(X, []) \\
&\phantom{\text{rev}\ X} \text{where } \text{rev}'(X, A) = A &&  \{|X| = 0\} \\
&\phantom{\text{rev}\ X\text{where } \text{rev}'(X, A)} = \text{rev}'(\text{tl}\ X, \text{hd}\ X:A) &&  \{|X| > 0\} \\
\text{take}(n, X) &= [] &&  \{n = 0\} \\
&= \text{hd}\ X:\text{take}(n-1,\text{tl}\ X) &&  \{n > 0\} \\
\text{drop}(n, X) &= X &&  \{n = 0\} \\
&= \text{hd}\ X:\text{drop}(n-1,\text{tl}\ X) &&  \{n > 0\}
\end{align*}
$$
其中，$\texttt{++}$ 和 $\text{take}$ 是增量的（incremental），即其中部分操作可以延迟到需要取出对应位置的元素时再执行；而 $\text{rev}$ 和 $\text{drop}$ 不是增量的，需要一次性将所有操作全部执行完成。

为了避免执行 $\text{rev}\ R$ 操作时的复杂度过高，我们需要提前执行该操作（而不是等到 $L$ 为空时再执行，此时 $|R|$ 太大了）。

周期性对 $\langle L,R\rangle$ 执行 $\langle{L\texttt{++}\text{rev}\ R, []}\rangle$，将这一操作称为 $\text{rot}$。

定义 $\text{rot}(L,R,[])=L\texttt{++}\text{rev}\ R$，用 $A$ 存储当前已经翻转的列表元素，则：
$$
\begin{align*}
\text{rot}(L,R,A)&=\text{hd}\ R:A && \{|L|=0\}\\
&=\text{hd}\ L:\text{rot}(\text{hd}\ L,\text{tl}\ R,\text{hd}\ R:A)&& \{|L|>0\}\\
\end{align*}
$$
由于 $\texttt{++}$ 是增量的，因此只要选择一个合适的周期，即可降低每次执行 $\text{rot}$ 操作时的复杂度。显然，当 $|R|=|L|+1$ 时最优。

如此，执行执行 $\text{remove}$ 时的最坏时间复杂度为 $O(\log{n})$（因为每次。

总结一下，操作函数为：
$$
\begin{alignat*}{3}
&[]_q &&= \langle [],[]\rangle&\\
&\langle L,R\rangle_q&&=|L|+|R|&\\
&\text{insert}(e,\langle L,R\rangle)&&=\text{makeq}\langle L, e:R\rangle\\
&\text{remove}\langle L,R\rangle&&=\langle\text{hd}\ L,\text{makeq}(\text{tl}\ L, R)\rangle\\
& \text{makeq}\langle L,R\rangle&&=\langle L,R\rangle&&&\{|R|\leq |L|\}\\
& &&=\langle\text{rot}(L,R,[]), []\rangle&&&\{|R|=|L|+1\}\\
&\text{rot}(L,R,A)&&=\text{hd}\ R:A &&& \{|L|=0\}\\
& &&=\text{hd}\ L:\text{rot}(\text{hd}\ L,\text{tl}\ R,\text{hd}\ R:A)&&& \{|L|>0\}\\
\end{alignat*}
$$

#### 代码实现

```haskell
data MyList a = MyList Int [a] deriving (Show)

-- 创建一个空列表
emptyList :: MyList a
emptyList = MyList 0 []

-- 添加元素到列表
append :: a -> MyList a -> MyList a
append x (MyList len xs) = MyList (len + 1) (x : xs)

-- 获取列表的长度
length' :: MyList a -> Int
length' (MyList len _) = len

-- 拼接两个 MyList
concatenate :: MyList a -> MyList a -> MyList a
concatenate (MyList len1 xs1) (MyList len2 xs2) =
  MyList (len1 + len2) (xs1 ++ xs2)

-- 获取前 n 个元素
take' :: Int -> MyList a -> MyList a
take' n (MyList len xs) = MyList (min n len) (take n xs)

rev :: MyList a -> MyList a
rev x = rev' x emptyList
  where
    rev' (MyList _ []) acc = acc
    rev' (MyList len (x:xs)) acc = rev' (MyList (len - 1) xs) (append x acc)

type PairedLazyList a = (MyList a, MyList a)
newtype Queue a = Queue (PairedLazyList a) deriving (Show)

emptyQueue :: Queue a
emptyQueue = Queue (emptyList, emptyList)

push :: a -> Queue a -> Queue a
push x (Queue (front, tail)) = makeq (Queue (front, append x tail))

front :: Queue a -> Maybe(a, Queue a)
front (Queue (MyList 0 _, _)) = Nothing
front (Queue (MyList len (x:xs), rear)) = Just (x, makeq (Queue (MyList len (x:xs), rear)))

pop :: Queue a -> Maybe(Queue a)
pop (Queue (MyList 0 _, _)) = Nothing
pop (Queue (MyList len (x:xs), rear)) = Just (makeq (Queue (MyList (len - 1) xs, rear)))

makeq :: Queue a -> Queue a
makeq (Queue (MyList lf xf, MyList lr xr))
    | lf >= lr = Queue (MyList lf xf, MyList lr xr)
    | otherwise = Queue (rot(MyList lf xf) (MyList lr xr) emptyList, emptyList)

rot :: MyList a -> MyList a -> MyList a -> MyList a
rot (MyList 0 _) r a = concatenate r a
rot (MyList ll (l:xl)) (MyList lr (r:xr)) a = append l (rot (MyList (ll-1) xl) (MyList (lr-1) xr) (append r a))

empty :: Queue a -> Bool
empty (Queue (MyList 0 _, _)) = True
empty _ = False

size :: Queue a -> Int
size (Queue (MyList ll _, MyList lr _)) = ll + lr
```

### With pre-evaluation

为了保证单次操作的最坏复杂度为 $O(1)$，我们需要在 $O(1)$ 的时间内得到当前查询的队尾元素。

因此我们需要提前预处理出可能的队尾元素，在需要时直接取出。

现在使用 $\langle L,R,\hat{L}\rangle$ 来表示 $q$，其中 $L,R$ 的含义与此前相同，而 $\hat{L}$ 存储的是 $L$ 的部分队尾元素，标志着 $L$ 的已评估部分和未评估部分之间的边界。

当 $\hat{L}=[]$ 时，整个列表的元素都已计算，而每当进行 $\text{insert}$ 和 $\text{remove}$ 操作（未触发 $\text{rot}$ ）时， 都会导致  $\hat{L}$ 移动一格，需要对下一个元素进行评估。而每次指定 $\text{rot}$ 操作后，将 $\hat{L}$ 置为 $L$。

我们需要保证每次执行 $\text{rot}$ 时，$\hat{L}=[]$。因此让 $|\hat{L}|=|L|-|R|$，当执行 $\text{rot}$ 时， $|L|=|R|$，此时正好 $|\hat{L}|=0$。

此时的操作函数为：
$$
\begin{alignat*}{3}
&[]_q &&= \langle [],[],[]\rangle&\\
&\langle L,R,\hat{L}\rangle_q&&=|L|+|R|&\\
&\text{insert}(e,\langle L,R,\hat{L}\rangle)&&=\text{makeq}\langle L, e:R,\hat{L}\rangle\\
&\text{remove}\langle L,R,\hat{L}\rangle&&=\langle\text{hd}\ L,\text{makeq}(\text{tl}\ L, R,\hat{L})\rangle\\
& \text{makeq}\langle L,R\rangle&&=\langle L,R,\text{tl}\ \hat{L}\rangle&&&\{|\hat{L}|> 0\}\\
& &&=\text{let}\ L'=\text{rot}(L,R,[])\ \text{in}\ \langle L',[],L'\rangle&&&\{|\hat{L}|=0\}\\
&\text{rot}(L,R,A)&&=\text{hd}\ R:A &&& \{|L|=0\}\\
& &&=\text{hd}\ L:\text{rot}(\text{hd}\ L,\text{tl}\ R,\text{hd}\ R:A)&&& \{|L|>0\}\\
\end{alignat*}
$$

#### 代码

需要特别注明的是，如果不需要查询队列的大小，则可以使用内置的List而非自定义MyList（因为不需要维护 $L,R,\hat{L}$ 的长度，只需要判断 $\hat{L}$ 是否为空）。

```haskell
import System.Exit (exitSuccess)
data MyList a = MyList Int [a] deriving (Show)

-- 创建一个空列表
emptyList :: MyList a
emptyList = MyList 0 []

-- 添加元素到列表
append :: a -> MyList a -> MyList a
append x (MyList len xs) = MyList (len + 1) (x : xs)

-- 获取列表的长度
length' :: MyList a -> Int
length' (MyList len _) = len

-- 拼接两个 MyList
concatenate :: MyList a -> MyList a -> MyList a
concatenate (MyList len1 xs1) (MyList len2 xs2) =
  MyList (len1 + len2) (xs1 ++ xs2)

-- 获取前 n 个元素
take' :: Int -> MyList a -> MyList a
take' n (MyList len xs) = MyList (min n len) (take n xs)

rev :: MyList a -> MyList a
rev x = rev' x emptyList
  where
    rev' (MyList _ []) acc = acc
    rev' (MyList len (x:xs)) acc = rev' (MyList (len - 1) xs) (append x acc)

type TripleLazyList a = (MyList a, MyList a, MyList a)
newtype Queue a = Queue (TripleLazyList a) deriving (Show)

emptyQueue :: Queue a
emptyQueue = Queue (emptyList, emptyList, emptyList)

push :: a -> Queue a -> Queue a
push x (Queue (front, tail, pp)) = makeq (Queue (front, append x tail, pp))

front :: Queue a -> Maybe(a, Queue a)
front (Queue (MyList 0 _, _, _)) = Nothing
front (Queue (MyList len xs, rear, pp)) = Just (head xs, Queue (MyList len xs, rear, pp))

pop :: Queue a -> Maybe(Queue a)
pop (Queue (MyList 0 _, _, _)) = Nothing
pop (Queue (MyList len (x:xs), rear, pp)) = Just (makeq (Queue (MyList (len - 1) xs, rear, pp)))

makeq :: Queue a -> Queue a
makeq (Queue (front, rear, MyList 0 _)) = Queue (newl, emptyList, newl) where newl = rot front rear emptyList
makeq (Queue (MyList lf front, rear, MyList lp (p:pp))) = Queue (MyList lf front, rear, MyList (lp - 1) pp)

rot :: MyList a -> MyList a -> MyList a -> MyList a
rot (MyList 0 _) r a = concatenate r a
rot (MyList ll (l:xl)) (MyList lr (r:xr)) a = append l (rot (MyList (ll-1) xl) (MyList (lr-1) xr) (append r a))

empty :: Queue a -> Bool
empty (Queue (MyList 0 _, _, _)) = True
empty _ = False

size :: Queue a -> Int
size (Queue (MyList ll _, MyList lr _, _)) = ll + lr 
```

## 备注

三份代码的正确性通过 [洛谷 B3616 【模板】队列](https://www.luogu.com.cn/problem/B3616) 验证，没想到洛谷如今已经支持了Haskell语言的提交评测。

根据题目要求，代码中的交互部分如下：

```haskell
import System.Exit (exitSuccess)
foldM' :: (Monad m) => (a -> b -> m a) -> a -> [b] -> m a 
foldM' _ z [] = return z 
foldM' f z (x:xs) = do 
    z' <- f z x 
    z' `seq` foldM' f z' xs 

opt :: Queue Int -> IO(Queue Int)
opt q = do
  input <- getLine
  let integers = map read (words input) :: [Int]
  case head integers of
    1 -> return $ push (integers !! 1) q
    2 -> case pop q of
      Nothing -> do
        putStrLn "ERR_CANNOT_POP"
        return q
      Just q2 -> return q2
    3 -> case front q of
      Just (x, q2) -> do
        print x
        return q2
      Nothing -> do
        putStrLn "ERR_CANNOT_QUERY"
        return q
    4 -> do
      print (size q)
      return q

main :: IO ()
main = do
  n <- readLn :: IO Int
  let initialQueue = (emptyQueue :: Queue Int)
  finalQueue <- foldM' (\q _ -> opt q) initialQueue [1..n]
  exitSuccess
```


