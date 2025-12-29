import streamlit as st
import math
import random
import itertools
from typing import Any, Dict, List, Optional, Set, Tuple
from streamlit_agraph import agraph, Node, Edge, Config
from collections import Counter

# Sayfa Ayarları
st.set_page_config(layout="wide", page_title="KTU Planar Solver")

# --- TİP TANIMLAMALARI ---
EdgeT = Tuple[int, int]
Rotation = Dict[int, List[int]]


# --- YARDIMCI FONKSİYONLAR ---

def canon_edge(u: int, v: int) -> EdgeT:
	"""Kenarı her zaman (küçük, büyük) formatına getirir."""
	return (u, v) if u < v else (v, u)


def safe_parse_clicked(clicked: Any) -> Optional[int]:
	"""Tıklanan node ID'sini güvenli bir şekilde int'e çevirir."""
	if clicked is None:
		return None
	if isinstance(clicked, dict) and "id" in clicked:
		try:
			return int(clicked["id"])
		except Exception:
			return None
	try:
		return int(clicked)
	except Exception:
		return None


def build_adj(nodes: List[int], edges: List[EdgeT]) -> Dict[int, Set[int]]:
	"""Komşuluk listesi oluşturur."""
	adj = {n: set() for n in nodes}
	for u, v in edges:
		u, v = canon_edge(u, v)
		if u == v:
			continue
		if u in adj and v in adj:
			adj[u].add(v)
			adj[v].add(u)
	return adj







def euler_bound_planar(v: int, e: int) -> bool:
	"""Euler formülüne göre temel planarlık kontrolü."""
	if v <= 2:
		return True
	return e <= 3 * v - 6


# ------------------------------------------------------
#  KURATOWSKI İPUCU: K5 / K3,3 ARAMA (SUBGRAPH + SMOOTHING)
# ------------------------------------------------------

def _find_K5(nodes: List[int], adj: Dict[int, Set[int]]) -> Optional[List[int]]:
	"""K5 altgrafı var mı? (extra kenarlar olabilir; biz sadece tüm çiftler bağlı mı bakarız)"""
	if len(nodes) < 5:
		return None
	for comb in itertools.combinations(nodes, 5):
		ok = True
		for u, v in itertools.combinations(comb, 2):
			if v not in adj[u]:
				ok = False
				break
		if ok:
			return list(comb)
	return None


def _find_K33(nodes: List[int], adj: Dict[int, Set[int]]) -> Optional[Tuple[List[int], List[int]]]:
	"""K3,3 altgrafı var mı? (extra kenarlar olabilir; cross-edge'ler tam mı bakarız)"""
	if len(nodes) < 6:
		return None

	for comb in itertools.combinations(nodes, 6):
		comb = list(comb)
		# 3-3 partition dene: A seç, B gerisi
		for A in itertools.combinations(comb, 3):
			A = set(A)
			B = [x for x in comb if x not in A]
			# A-B arasında tüm kenarlar var mı?
			ok = True
			for a in A:
				for b in B:
					if b not in adj[a]:
						ok = False
						break
				if not ok:
					break
			if ok:
				return (sorted(list(A)), sorted(B))
	return None


def _smooth_degree2(nodes: List[int], edges: List[EdgeT]) -> Tuple[List[int], List[EdgeT]]:
	"""
	Derecesi 2 olan düğümleri 'eriterek' homeomorphic core çıkarır:
	u - x - v => u - v, x silinir.
	(Not: Bu Kuratowski subdivision yakalamada pratik bir yaklaşımdır.)
	"""
	nodes = list(nodes)
	edges = [canon_edge(u, v) for (u, v) in edges if u != v]
	adj = build_adj(nodes, edges)

	changed = True
	while changed:
		changed = False
		for x in list(nodes):
			if x not in adj:
				continue
			if len(adj[x]) == 2:
				u, v = list(adj[x])
				# x'i kaldır, u-v bağla
				adj[u].discard(x)
				adj[v].discard(x)
				adj.pop(x, None)
				if x in nodes:
					nodes.remove(x)

				if u != v:
					adj.setdefault(u, set()).add(v)
					adj.setdefault(v, set()).add(u)

				changed = True
				break

	# adj'den edge listesine dön
	new_edges: Set[EdgeT] = set()
	for u in nodes:
		for v in adj.get(u, set()):
			if u < v:
				new_edges.add((u, v))
	return sorted(nodes), sorted(list(new_edges))


def kuratowski_hint(nodes: List[int], edges: List[EdgeT]) -> Optional[str]:
	"""
	Nonplanar çıkınca kullanıcıya 'K5 mi K3,3 mü' konusunda ipucu üretir.
	Önce ham graf üzerinde bakar, sonra degree-2 smoothing ile çekirdekte bakar.
	"""
	nodes = sorted(nodes)
	edges = sorted(list(set(canon_edge(u, v) for u, v in edges if u != v)))
	adj = build_adj(nodes, edges)

	k5 = _find_K5(nodes, adj)
	if k5 is not None:
		return f"Kuratowski ipucu: **K5** bulundu (düğümler={k5})"

	k33 = _find_K33(nodes, adj)
	if k33 is not None:
		A, B = k33
		return f"Kuratowski ipucu: **K3,3** bulundu (A={A}, B={B})"

	# smoothing dene
	sn, se = _smooth_degree2(nodes, edges)
	sadj = build_adj(sn, se)

	k5s = _find_K5(sn, sadj)
	if k5s is not None:
		return f"Kuratowski ipucu (smoothing sonrası): **K5** bulundu (düğümler={k5s})"

	k33s = _find_K33(sn, sadj)
	if k33s is not None:
		A, B = k33s
		return f"Kuratowski ipucu (smoothing sonrası): **K3,3** bulundu (A={A}, B={B})"

	return None


# --- PLANARLIK TESTİ  ---

def spanning_tree(nodes: List[int], edges: List[EdgeT]) -> List[EdgeT]:
	adj = build_adj(nodes, edges)
	seen: Set[int] = set()
	tree: List[EdgeT] = []
	for s in nodes:
		if s in seen:
			continue
		seen.add(s)
		q = [s]
		while q:
			x = q.pop(0)
			for y in sorted(adj[x]):
				if y not in seen:
					seen.add(y)
					q.append(y)
					tree.append(canon_edge(x, y))
	return tree


def init_tree_rotation(nodes: List[int], tree_edges: List[EdgeT]) -> Rotation:
	rot: Rotation = {v: [] for v in nodes}
	for u, v in tree_edges:
		u, v = canon_edge(u, v)
		rot[u].append(v)
		rot[v].append(u)
	for v in rot:
		rot[v].sort()
	return rot


def next_dart(rot: Rotation, u: int, v: int) -> Tuple[int, int]:
	nbrs = rot[v]
	i = nbrs.index(u)
	w = nbrs[(i - 1) % len(nbrs)]
	return (v, w)


def faces_from_rotation(rot: Rotation) -> List[List[Tuple[int, int]]]:
	used: Set[Tuple[int, int]] = set()
	faces: List[List[Tuple[int, int]]] = []
	for v in rot:
		for u in rot[v]:
			d = (v, u)
			if d in used:
				continue
			cycle: List[Tuple[int, int]] = []
			cur = d
			while cur not in used:
				used.add(cur)
				cycle.append(cur)
				cur = next_dart(rot, cur[0], cur[1])
			faces.append(cycle)
	return faces


def face_vertices(face_darts: List[Tuple[int, int]]) -> List[int]:
	return [u for (u, _) in face_darts]


def corners_on_face(face: List[Tuple[int, int]], v: int) -> List[Tuple[int, int]]:
	corners: List[Tuple[int, int]] = []
	m = len(face)
	for i in range(m):
		a_u = face[i]
		u_b = face[(i + 1) % m]
		if a_u[1] == v and u_b[0] == v:
			corners.append((a_u[0], u_b[1]))
	return corners


def insert_before(lst: List[int], target: int, newval: int) -> List[int]:
	i = lst.index(target)
	return lst[:i] + [newval] + lst[i:]


def try_insert_edge(rot: Rotation, u: int, v: int, face: List[Tuple[int, int]],
                    cu: Tuple[int, int], cv: Tuple[int, int]) -> Optional[Rotation]:
	if v in rot[u] or u in rot[v]:
		return None
	a, _ = cu
	c, _ = cv
	if a not in rot[u] or c not in rot[v]:
		return None
	new_rot = {x: rot[x][:] for x in rot}
	new_rot[u] = insert_before(new_rot[u], a, v)
	new_rot[v] = insert_before(new_rot[v], c, u)
	return new_rot



def is_planar_exact(nodes: List[int], edges: List[EdgeT]) -> bool:
    # Planarsa embedding (rotation) bulunur, değilse None döner
    return find_planar_rotation(nodes, edges, seed=97) is not None

def find_planar_rotation(nodes: List[int], edges: List[EdgeT], seed: int = 97) -> Optional[Rotation]:
	"""
	Planarsa -> Rotation system (embedding)
	Nonplanarsa -> None
	Not: Bu fonksiyon embedding_search_exact'in aynısını yapar, sadece başarılı olunca rot döndürür.
	"""
	nodes = sorted(nodes)
	edges = sorted(list(set(canon_edge(u, v) for u, v in edges if u != v)))

	if not euler_bound_planar(len(nodes), len(edges)):
		return None
	if not edges or len(edges) == 1:
		# trivially planar; basit bir rot üretelim
		tree = spanning_tree(nodes, edges)
		return init_tree_rotation(nodes, tree)

	tree = spanning_tree(nodes, edges)
	tree_set = set(tree)
	rem = [e for e in edges if e not in tree_set]
	rot0 = init_tree_rotation(nodes, tree)

	rnd = random.Random(seed)
	deg = {v: 0 for v in nodes}
	for u, v in edges:
		deg[u] += 1
		deg[v] += 1
	rem.sort(key=lambda e: -(deg[e[0]] + deg[e[1]]))

	def options_count(rot: Rotation, e: EdgeT) -> int:
		u, v = e
		cnt = 0
		for f in faces_from_rotation(rot):
			vs = face_vertices(f)
			if u in vs and v in vs:
				cu = corners_on_face(f, u)
				cv = corners_on_face(f, v)
				cnt += len(cu) * len(cv)
		return cnt

	def backtrack(rot: Rotation, rem_edges: List[EdgeT]) -> Optional[Rotation]:
		if not rem_edges:
			return rot  # ✅ embedding bulundu

		best_i = -1
		best_cnt = 10**9
		for i, e in enumerate(rem_edges):
			c = options_count(rot, e)
			if c == 0:
				return None
			if c < best_cnt:
				best_cnt = c
				best_i = i
				if best_cnt == 1:
					break

		e = rem_edges[best_i]
		rest = rem_edges[:best_i] + rem_edges[best_i + 1:]
		u, v = e

		moves = []
		for f in faces_from_rotation(rot):
			vs = face_vertices(f)
			if u in vs and v in vs:
				cu = corners_on_face(f, u)
				cv = corners_on_face(f, v)
				for cu1 in cu:
					for cv1 in cv:
						moves.append((f, cu1, cv1))
		rnd.shuffle(moves)

		for f, cu1, cv1 in moves:
			new_rot = try_insert_edge(rot, u, v, f, cu1, cv1)
			if new_rot is None:
				continue
			res = backtrack(new_rot, rest)
			if res is not None:
				return res

		return None

	return backtrack(rot0, rem)

def embedding_search_with_blame(nodes: List[int], edges: List[EdgeT], seed: int) -> Optional[EdgeT]:
	"""
	Planarsa -> None
	Nonplanarsa -> embedding sırasında ilk "yüze oturamayan" kenarı döndürür (blame edge).
	Not: Bu bir heuristik sertifika gibi kullanılır.
	"""
	nodes = sorted(nodes)
	edges = sorted(list(set(canon_edge(u, v) for u, v in edges if u != v)))

	if not euler_bound_planar(len(nodes), len(edges)):
		return None

	if len(edges) <= 1:
		return None

	tree = spanning_tree(nodes, edges)
	tree_set = set(tree)
	rem = [e for e in edges if e not in tree_set]
	rot0 = init_tree_rotation(nodes, tree)

	rnd = random.Random(seed)

	def options_count(rot: Rotation, e: EdgeT) -> int:
		u, v = e
		cnt = 0
		for f in faces_from_rotation(rot):
			vs = face_vertices(f)
			if u in vs and v in vs:
				cu = corners_on_face(f, u)
				cv = corners_on_face(f, v)
				cnt += len(cu) * len(cv)
		return cnt

	def backtrack(rot: Rotation, rem_edges: List[EdgeT]) -> Optional[EdgeT]:
		if not rem_edges:
			return None

		# MRV: en az seçenekli kenarı seçelim (senin exact fonksiyonundaki mantığa yakın)
		best_i = -1
		best_cnt = 10**9
		for i, e in enumerate(rem_edges):
			c = options_count(rot, e)
			if c == 0:
				return e  # 🔥 suçlu kenar
			if c < best_cnt:
				best_cnt = c
				best_i = i
				if best_cnt == 1:
					break

		e = rem_edges[best_i]
		rest = rem_edges[:best_i] + rem_edges[best_i + 1:]
		u, v = e

		moves = []
		for f in faces_from_rotation(rot):
			vs = face_vertices(f)
			if u in vs and v in vs:
				cu = corners_on_face(f, u)
				cv = corners_on_face(f, v)
				for cu1 in cu:
					for cv1 in cv:
						moves.append((f, cu1, cv1))
		rnd.shuffle(moves)

		for f, cu1, cv1 in moves:
			new_rot = try_insert_edge(rot, u, v, f, cu1, cv1)
			if new_rot is None:
				continue
			b = backtrack(new_rot, rest)
			if b is None:
				return None  # planar bulundu
			if b:
				return b

		# burada başarısızsak, bu kenar da güçlü adaydır
		return e

	return backtrack(rot0, rem)


def find_critical_edge(nodes: List[int], edges: List[EdgeT], trials: int = 25) -> Optional[EdgeT]:
	"""
	Farklı seed'lerle embedding dener.
	Nonplanarlıkta dönen blame-edge'leri sayar.
	En sık geçen kenarı 'kritik' kabul eder.
	"""
	if not edges:
		return None

	cnt = Counter()
	for i in range(trials):
		b = embedding_search_with_blame(nodes, edges, seed=100 + 31 * i)
		if b is not None:
			cnt[canon_edge(b[0], b[1])] += 1

	if not cnt:
		return None
	return cnt.most_common(1)[0][0]



# --- GEOMETRİK VE ÇİZİM FONKSİYONLARI ---

def circle_layout(nodes: List[int], radius: float) -> Dict[int, List[float]]:
	if not nodes:
		return {}
	n = len(nodes)
	return {
		v: [radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n)]
		for i, v in enumerate(nodes)
	}


def _orient(ax, ay, bx, by, cx, cy):
	return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(ax, ay, bx, by, cx, cy):
	return min(ax, bx) <= cx <= max(ax, bx) and min(ay, by) <= cy <= max(ay, by)


def segments_intersect(p1, p2, q1, q2) -> bool:
	if p1 == q1 or p1 == q2 or p2 == q1 or p2 == q2:
		return False
	ax, ay = p1
	bx, by = p2
	cx, cy = q1
	dx, dy = q2
	o1 = _orient(ax, ay, bx, by, cx, cy)
	o2 = _orient(ax, ay, bx, by, dx, dy)
	o3 = _orient(cx, cy, dx, dy, ax, ay)
	o4 = _orient(cx, cy, dx, dy, bx, by)
	if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
		return True
	eps = 1e-12
	if abs(o1) < eps and _on_segment(ax, ay, bx, by, cx, cy): return True
	if abs(o2) < eps and _on_segment(ax, ay, bx, by, dx, dy): return True
	if abs(o3) < eps and _on_segment(cx, cy, dx, dy, ax, ay): return True
	if abs(o4) < eps and _on_segment(cx, cy, dx, dy, bx, by): return True
	return False


def crossing_count(pos: Dict[int, List[float]], edges: List[EdgeT]) -> int:
	es = [canon_edge(u, v) for (u, v) in edges if u != v]
	m = len(es)
	cnt = 0
	for i in range(m):
		u1, v1 = es[i]
		p1 = (pos[u1][0], pos[u1][1])
		p2 = (pos[v1][0], pos[v1][1])
		for j in range(i + 1, m):
			u2, v2 = es[j]
			if u1 in (u2, v2) or v1 in (u2, v2):
				continue
			q1 = (pos[u2][0], pos[u2][1])
			q2 = (pos[v2][0], pos[v2][1])
			if segments_intersect(p1, p2, q1, q2):
				cnt += 1
	return cnt


def layout_greedy(nodes: List[int], edges: List[EdgeT]) -> Dict[int, List[float]]:
	"""Kesişmeyi azaltmaya çalışan greedy/force hibriti."""
	if not nodes:
		return {}

	nodes = list(nodes)
	edges = [canon_edge(u, v) for (u, v) in edges if u != v]
	adj = build_adj(nodes, edges)

	# Başlangıç pozisyonu
	pos = circle_layout(nodes, radius=520.0)
	rnd = random.Random(123)
	for v in nodes:
		pos[v][0] += rnd.uniform(-20, 20)
		pos[v][1] += rnd.uniform(-20, 20)

	# En çok komşusu olanları önce yerleştir
	order = sorted(nodes, key=lambda v: len(adj[v]), reverse=True)

	# Arama yarıçapları
	radii = [80.0, 140.0, 220.0, 320.0, 460.0]
	angles = [2.0 * math.pi * k / 24.0 for k in range(24)]

	def spread(steps: int = 14):
		eps = 1e-6
		for _ in range(steps):
			fx = {v: 0.0 for v in nodes}
			fy = {v: 0.0 for v in nodes}
			for i in range(len(nodes)):
				a = nodes[i]
				for j in range(i + 1, len(nodes)):
					b = nodes[j]
					dx = pos[a][0] - pos[b][0]
					dy = pos[a][1] - pos[b][1]
					d2 = dx * dx + dy * dy + eps
					f = 9000.0 / d2
					fx[a] += f * dx
					fy[a] += f * dy
					fx[b] -= f * dx
					fy[b] -= f * dy
			for v in nodes:
				pos[v][0] += 0.8 * fx[v]
				pos[v][1] += 0.8 * fy[v]

	for _pass in range(10):
		improved = False
		base_cross = crossing_count(pos, edges)

		for v in order:
			cx, cy = pos[v]
			if adj[v]:
				mx = sum(pos[u][0] for u in adj[v]) / len(adj[v])
				my = sum(pos[u][1] for u in adj[v]) / len(adj[v])
			else:
				mx = sum(pos[u][0] for u in nodes) / len(nodes)
				my = sum(pos[u][1] for u in nodes) / len(nodes)

			mx += rnd.uniform(-15, 15)
			my += rnd.uniform(-15, 15)

			candidates = [(cx, cy), (mx, my)]
			for r in radii:
				for a in angles:
					candidates.append((mx + r * math.cos(a), my + r * math.sin(a)))
			for r in [80.0, 140.0, 220.0]:
				for a in angles:
					candidates.append((cx + r * math.cos(a), cy + r * math.sin(a)))

			best = (cx, cy)
			best_cross = base_cross

			for px, py in candidates:
				pos[v] = [px, py]
				c = crossing_count(pos, edges)
				if c < best_cross:
					best_cross = c
					best = (px, py)
					if best_cross == 0:
						break

			if best_cross < base_cross:
				pos[v] = [best[0], best[1]]
				base_cross = best_cross
				improved = True
			else:
				pos[v] = [cx, cy]

		spread()

		if crossing_count(pos, edges) == 0:
			break
		if not improved:
			break

	return pos
def _outer_face_cycle(rot: Rotation) -> List[int]:
	"""
	Rotation'dan yüzleri çıkarır, en uzun yüzü 'outer face' varsayar
	ve o yüzün düğümlerini çevrim sırasıyla döndürür.
	"""
	faces = faces_from_rotation(rot)
	if not faces:
		return []
	# en uzun yüz (dart sayısı en büyük)
	outer = max(faces, key=lambda f: len(f))
	cyc = face_vertices(outer)

	# ardışık tekrarları temizle (bazı durumlarda aynı düğüm art arda gelebilir)
	out = []
	for v in cyc:
		if not out or out[-1] != v:
			out.append(v)
	# kapalı çevrimde baş=son olabilir; onu da sadeleştir
	if len(out) >= 2 and out[0] == out[-1]:
		out.pop()
	return out


def layout_from_embedding_tutte(nodes: List[int], edges: List[EdgeT], rot: Rotation,
                               R: float = 520.0, iters: int = 600, alpha: float = 0.85) -> Dict[int, List[float]]:
	"""
	Embedding (rotation system) üzerinden çizim:
	1) Outer face düğümleri çembere sabitlenir.
	2) Diğer düğümler için barycentric (komşu ortalaması) iterasyonu yapılır.
	   x_v <- (1-alpha)*x_v + alpha*avg_{u in N(v)} x_u
	"""
	nodes = sorted(nodes)
	edges = [canon_edge(u, v) for (u, v) in edges if u != v]
	adj = build_adj(nodes, edges)

	outer = _outer_face_cycle(rot)
	if len(outer) < 3:
		# outer yüz bulamazsak fallback
		return layout_greedy(nodes, edges)

	outer_set = set(outer)

	# 1) Outer face çemberde sabit
	pos: Dict[int, List[float]] = {}
	m = len(outer)
	for i, v in enumerate(outer):
		theta = 2.0 * math.pi * i / m
		pos[v] = [R * math.cos(theta), R * math.sin(theta)]

	# 2) İç düğümler için başlangıç (küçük çember + gürültü)
	rnd = random.Random(1234)
	for v in nodes:
		if v in outer_set:
			continue
		theta = rnd.random() * 2.0 * math.pi
		rr = 0.35 * R
		pos[v] = [rr * math.cos(theta) + rnd.uniform(-15, 15),
		          rr * math.sin(theta) + rnd.uniform(-15, 15)]

	# 3) Barycentric iterasyon: outer sabit, iç düğümler komşu ortalamasına çekilir
	inner = [v for v in nodes if v not in outer_set and len(adj[v]) > 0]
	if not inner:
		return pos

	for _ in range(iters):
		max_move = 0.0
		for v in inner:
			nbrs = list(adj[v])
			if not nbrs:
				continue
			ax = sum(pos[u][0] for u in nbrs) / len(nbrs)
			ay = sum(pos[u][1] for u in nbrs) / len(nbrs)

			# damping / relaxation
			nx = (1.0 - alpha) * pos[v][0] + alpha * ax
			ny = (1.0 - alpha) * pos[v][1] + alpha * ay

			dx = nx - pos[v][0]
			dy = ny - pos[v][1]
			pos[v][0] = nx
			pos[v][1] = ny

			mv = abs(dx) + abs(dy)
			if mv > max_move:
				max_move = mv

		# yakınsama kriteri (çok küçük hareket)
		if max_move < 1e-3:
			break

	return pos


# --- STREAMLIT ARAYÜZÜ ---

if "nodes" not in st.session_state:
	st.session_state["nodes"] = [1, 2, 3, 4]
if "edges" not in st.session_state:
	st.session_state["edges"] = []
if "sel" not in st.session_state:
	st.session_state["sel"] = None
if "mode" not in st.session_state:
	st.session_state["mode"] = "circle"
if "res" not in st.session_state:
	st.session_state["res"] = (True, "Çizime başlayın.")
if "last_edge" not in st.session_state:
	st.session_state["last_edge"] = None
if "critical_edge" not in st.session_state:
	st.session_state["critical_edge"] = None  # kırmızı gösterilecek kenar
if "pending_delete" not in st.session_state:
	st.session_state["pending_delete"] = False  # ikinci basışta sil


st.title("KTU Planar Solver")

# Anlık planarlık durumu (üst bilgi için)
planar_now = is_planar_exact(st.session_state["nodes"], st.session_state["edges"])

# 6 Sütunlu Kontrol Paneli
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
	if st.button("Düğüm Ekle"):
		nid = max(st.session_state["nodes"]) + 1 if st.session_state["nodes"] else 1
		st.session_state["nodes"].append(nid)
		st.session_state["mode"] = "circle"
		st.session_state["res"] = (True, "Düğüm eklendi.")
		st.rerun()

with c2:
	if st.button("Seçiliyi Sil"):
		if st.session_state["sel"] is not None:
			s = st.session_state["sel"]
			if s in st.session_state["nodes"]:
				st.session_state["nodes"].remove(s)
			st.session_state["edges"] = [e for e in st.session_state["edges"] if s not in e]
			st.session_state["sel"] = None
			st.session_state["last_edge"] = None
			st.session_state["mode"] = "circle"
			st.session_state["res"] = (True, "Düğüm silindi.")
			st.rerun()

with c3:
	if st.button("Graf Planar mı?"):
		planar = is_planar_exact(st.session_state["nodes"], st.session_state["edges"])
		if planar:
			st.session_state["res"] = (True, "Graf planar.")
		else:
			hint = kuratowski_hint(st.session_state["nodes"], st.session_state["edges"])
			msg = "Graf nonplanar."
			if hint:
				msg += f" ({hint})"
			st.session_state["res"] = (False, msg)
		st.rerun()

with c4:
	if st.button("Planar Yap", disabled=planar_now):

		# 2. basış: kırmızı gösterilen kenarı gerçekten sil
		if st.session_state.get("pending_delete") and st.session_state.get("critical_edge") in st.session_state["edges"]:
			e = st.session_state["critical_edge"]
			st.session_state["edges"].remove(e)
			st.session_state["pending_delete"] = False
			st.session_state["last_edge"] = None

			planar = is_planar_exact(st.session_state["nodes"], st.session_state["edges"])
			if planar:
				st.session_state["res"] = (True, f"Kenar silindi: {e}. Artık planar.")
				st.session_state["critical_edge"] = None
			else:
				hint = kuratowski_hint(st.session_state["nodes"], st.session_state["edges"])
				msg = f"Kenar silindi: {e}. Hâlâ nonplanar."
				if hint:
					msg += f" ({hint})"
				st.session_state["res"] = (False, msg)

			st.session_state["mode"] = "circle"
			st.rerun()

		# 1. basış: kritik kenarı bul ve kırmızıya boya (silme yok)
		e = find_critical_edge(st.session_state["nodes"], st.session_state["edges"], trials=25)

		# fallback: kritik bulunamazsa eski davranışa dön
		if e is None:
			e = st.session_state["last_edge"] if st.session_state["last_edge"] in st.session_state["edges"] else st.session_state["edges"][-1]

		st.session_state["critical_edge"] = e
		st.session_state["pending_delete"] = True
		st.session_state["res"] = (True, f"Kırmızı kenar silinmek üzere seçildi: {e}. Silmek için tekrar 'Planar Yap'a bas.")

		st.session_state["mode"] = "circle"
		st.rerun()


with c5:
	if st.button("Planar Çiz"):
		st.session_state["mode"] = "planar"
		st.rerun()

with c6:
	if st.button("Kenarları Temizle"):
		st.session_state["edges"] = []
		st.session_state["sel"] = None
		st.session_state["last_edge"] = None
		st.session_state["mode"] = "circle"
		st.session_state["res"] = (True, "Tüm kenarlar silindi.")
		st.rerun()

# --- MESAJ GÖSTERİMİ ---
ok, msg = st.session_state["res"]
if ok:
	st.success(msg)
else:
	st.error(msg)

# --- ÇİZİM ALANI ---

if st.session_state["mode"] == "planar":
	# 1) önce embedding bulmaya çalış
	rot = find_planar_rotation(st.session_state["nodes"], st.session_state["edges"], seed=97)

	if rot is not None:
		# 2) embedding tabanlı (outer face sabit + barycentric) çizim
		coords = layout_from_embedding_tutte(st.session_state["nodes"], st.session_state["edges"], rot)
	else:
		# 3) embedding yoksa fallback
		coords = layout_greedy(st.session_state["nodes"], st.session_state["edges"])
else:
	coords = circle_layout(st.session_state["nodes"], 350.0)

nodes_vis = []
for n in st.session_state["nodes"]:
	x, y = coords.get(n, [0.0, 0.0])
	nodes_vis.append(
		Node(
			id=n,
			label=str(n),
			x=float(x),
			y=float(y),
			size=25,
			color="#FFD700" if n == st.session_state["sel"] else "#97C2FC",
		)
	)

edges_vis = []
crit = st.session_state.get("critical_edge")
for u, v in st.session_state["edges"]:
	u, v = canon_edge(u, v)
	col = "red" if (crit is not None and canon_edge(crit[0], crit[1]) == (u, v)) else "black"
	edges_vis.append(Edge(source=u, target=v, color=col))


config = Config(width="100%", height=550, directed=False, physics={"enabled": False})
clicked = agraph(nodes=nodes_vis, edges=edges_vis, config=config)

# --- ETKİLEŞİM MANTIĞI ---

cid = safe_parse_clicked(clicked)
if cid is not None:
	if st.session_state["sel"] is None:
		st.session_state["sel"] = cid
	elif st.session_state["sel"] == cid:
		st.session_state["sel"] = None
	else:
		# İkinci tıklama: Kenar ekle veya sil
		e = canon_edge(st.session_state["sel"], cid)
		if e in st.session_state["edges"]:
			st.session_state["edges"].remove(e)
			st.session_state["critical_edge"] = None
			st.session_state["pending_delete"] = False
			st.session_state["res"] = (True, "Kenar silindi.")

		else:
			st.session_state["edges"].append(e)
			st.session_state["last_edge"] = e
			st.session_state["critical_edge"] = None
			st.session_state["pending_delete"] = False
			st.session_state["res"] = (True, "Kenar eklendi.")

		st.session_state["sel"] = None
		st.session_state["mode"] = "circle"  # Kenar değişince default'a dön
	st.rerun()
