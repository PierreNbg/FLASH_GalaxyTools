import h5py
import numpy as np

class Node:
    children = []
    neighbours = []
    parent = -1
    
    def __init__(self, bid, bbox, bs, rlvl):
        self.bid = bid
        self.bbox = bbox
        self.bs = bs
        self.rlvl = rlvl
        
    def setNeighbours(self, neighNodes):
        self.neighbours = neighNodes
    
    def setParent(self, parentNode):
        self.parent = parentNode
        
    def setChildren(self, childNodes):
        self.children = childNodes

    def getBid(self):
        return self.bid

    def isInside(self, pos, res=None):
        
        if res is None:
            res = np.zeros((pos.shape[0], 4), dtype=int)
            res[:, 0] = 0
        
        isOut = np.any(pos < self.bbox[:, 0], axis=1)
        isOut += np.any(pos >= self.bbox[:, 1], axis=1)
        
        if np.all(isOut):
            return res
        
        if len(self.children) > 0:
            for c in self.children:
                res += c.isInside(pos=pos)
            return res
        
        sel = np.where(np.logical_not(isOut))
        
        res[sel, 0] = self.bid
        cid = np.floor(
            (pos[sel] - self.bbox[:, 0][None, :]) / self.bs[None, :] * 8
        ).astype(int)
        res[sel, 1:4] += cid
        
        return res


pf = h5py.File("DwarfGal_hdf5_plt_cnt_0002", 'r')

gid = pf['gid'][()]
bbox = pf['bounding box'][()]
bs = pf['block size'][()]
rlvl = pf['refine level'][()]
dens = pf['dens'][()]

root_gids = np.where(gid[:, 6] == -1)
bbox_root = np.zeros_like(bbox[0])
bbox_root[:, 0] = bbox[0, :, 0]
bbox_root[:, 1] = bbox[-1, :, 1]

# Create root nodes
rootNode = Node(-5, bbox=bbox_root, bs=0.0, rlvl=0)
rootNode.setChildren([
    Node(
        bid=i+1, bbox=bbox[i],
        bs=bs[i], rlvl=rlvl[i]
    ) for i in root_gids[0]
])

# Create tree for current node
stack = [rootNode]
while len(stack) > 0:
    curNode = stack.pop()
    cid = curNode.getBid()
    if cid == -5:
        stack.extend(curNode.children)
        continue

    cgid = gid[cid-1]
    n = cgid[:6]
    p = cgid[6]
    c = cgid[7:]
    if np.any(c > 0):
        childNodes = []
        for jj in c:
            cNode = Node(bid=jj, bbox=bbox[jj-1], bs=bs[jj-1], rlvl=rlvl[0])
            # Set parent of child
            cNode.setParent(curNode)
            # Store child in list
            childNodes.append(cNode)
            # Add to stack
            stack.append(cNode)

        # Add children nodes to current node
        curNode.setChildren(childNodes)
        
pos = np.random.randn(3000000).reshape(1000000, 3)
rootNode.isInside(pos)


class Block:
    
    children = []
    
    def __init__(self, bid, bbox, bs, data):
        self.rlvl = 1
        self.bid = bid
        self.bbox = bbox
        self.bs = bs
        self.data = data
    
    def add_children(self, cids, data):
        childList = []
        for i in cids:
            bbox = self.bbox
            quad = np.asarray([i%2, (i//2)%2, (i//4)%2])
            bbox[:, 0] += quad * self.bs / 2.0
            bbox[:, 1] -= (1 - quad * self.bs / 2.0)
            childList.append(
                BlockTree(
                    bid=cids[i], bbox=bbox, bs=self.bs / 2.0,
                    data=data
                )
            )

        self.children = childList
        
    def
    
    
    

class Block:
    
    def __init__(self, gid, rlvl, bid, bs, data):
        # Initialize block properties
        self.bid = bid
        self.parent = gid[bid, 6]
        self.children = gid[bid, 7:]
        self.neighbours = gid[bid, :6]
        self.level = rlvl[bid]
        self.data = data[bid]
        self.cs = bs[bid] / self.data.shape
    
    def add_gc(self):
        self.data = np.pad(
            self.data,
            pad_width=(
                (1, 1),
                (1, 1),
                (1, 1)
            )
        )
    
    def rm_gc(self):
        self.data = self.data[1:-1, 1:-1, 1:-1]
    
    # Get neighbour of parent if no neighbour at current level is available
    def find_neigh(self, gid_glob):
        i = 0
        for it in self.neighbours:

            if it == -1 - 1:
                par = self.parent
                self.neighbours[i] = gid_glob[par, i]
            i += 1
    
    
    
    def fill_gc(self, data_glob):
        i = 0
        for it in self.neighbours:
            sl_up = [slice(None), slice(None), slice(None)]
            
            # 0 -- negative neighbour, 1 -- positive neighbour
            n_mp = i % 2
            # 2 -- x, 1 -- y, 0 -- z
            xyz = 2 - i // 3
            if n_mp:
                sl_up[xyz] = slice(-1, None)
            else:
                sl_up[xyz] = slice(0, 1)
            
            sl_n
            
            i += 1
