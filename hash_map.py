class hash_map:

    def __init__(self, size) -> None:
        self.size = size
        self.map = [[] for i in range(size)]

    def _hash(self, key):
        return hash(key)%self.size
        
    def insert(self, key, value):
        hashed_key = self._hash(key)
        for _ in self.map[hashed_key]:
            if _[0] == key:
                if _[1] == value:
                    print("already existing key:value pair")
        print(f"successfully added [{key}, {value}]")
        return self.map[hashed_key].append([key, value])
    
    def get(self, key):
        hashed_key = self._hash(key)
        print(f"{hashed_key} retrieved")
        for _ in self.map[hashed_key]:
            if _[0] == key:
                print(f"value of key retrieved: {_[1]}")
                return _[1]
            
        return None



my_dict = hash_map(3)

my_dict.insert("test", "girl")
my_dict.insert("banana", 20)
value = my_dict.get("test")

print()