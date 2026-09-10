/* The fleet allocation, reimplemented in JavaScript so the demo runs with no backend.
 *
 * Mirrors src/allocate.py. tools/check_js_matches_python.py asserts the two produce
 * identical assignments and identical expected revenue on the real cell table.
 */

// Capacity: how many drivers a cell can absorb in an hour, derived from its own observed
// demand rather than assumed. Without it, top-k sends the whole fleet to one cell.
function capacityOf(tripsPerHour, maxMultiple){
  return Math.max(1, Math.floor(tripsPerHour * (maxMultiple || 1.5)));
}

// Bounded min-heap top-k: O(n log k) time, O(k) memory. The heap holds the k best seen so
// far with the WORST at the root, so a candidate that cannot beat the root is discarded in
// O(1) without ever entering the heap.
function topK(items, k, key){
  if(k <= 0) return [];
  const heap = [];
  const up = i => { while(i > 0){ const p = (i-1) >> 1;
      if(heap[p][0] <= heap[i][0]) break; [heap[p],heap[i]]=[heap[i],heap[p]]; i = p; } };
  const down = i => { for(;;){ const l=2*i+1, r=l+1; let s=i;
      if(l < heap.length && heap[l][0] < heap[s][0]) s = l;
      if(r < heap.length && heap[r][0] < heap[s][0]) s = r;
      if(s === i) break; [heap[s],heap[i]]=[heap[i],heap[s]]; i = s; } };

  let counter = 0;   // tie-breaker: equal keys must not compare the payload objects
  for(const item of items){
    const v = key(item);
    if(heap.length < k){ heap.push([v, counter++, item]); up(heap.length-1); }
    else if(v > heap[0][0]){ heap[0] = [v, counter++, item]; down(0); }
  }
  return heap.sort((a,b) => b[0]-a[0]).map(e => e[2]);
}

// Greedy capacity-constrained fill. Provably optimal here: every driver-hour is identical
// and independent, and capacities are hard constraints.
function allocate(cells, fleet, hour, dow, policy, maxMultiple){
  maxMultiple = maxMultiple || 1.5;
  const candidates = cells.filter(c => c[2] === hour && c[1] === dow &&
    (policy === "trip_volume" ? c[3] > 0 : c[4] > 0));

  let ranked;
  if(policy === "uniform"){
    ranked = candidates;
  } else {
    const key = policy === "trip_volume" ? (c => c[3]) : (c => c[4]);
    // Fetch more cells than drivers: a high-value cell may absorb only a handful, so a
    // top-`fleet` fetch can run out of capacity before it runs out of drivers.
    ranked = topK(candidates, Math.min(candidates.length, Math.max(fleet,1) * 4), key);
  }

  const assignments = [];
  let remaining = fleet, revenue = 0;

  if(policy === "uniform"){
    const per = Math.max(1, Math.floor(fleet / Math.max(1, candidates.length)));
    for(const c of candidates){
      if(remaining <= 0) break;
      const drivers = Math.min(remaining, per);
      revenue += drivers * c[4]; remaining -= drivers;
      assignments.push({zone:c[0], dow:c[1], hour:c[2], drivers, rph:c[4],
                        capacity:capacityOf(c[3],maxMultiple)});
    }
  } else {
    for(const c of ranked){
      if(remaining <= 0) break;
      const cap = capacityOf(c[3], maxMultiple);
      const drivers = Math.min(remaining, cap);
      if(drivers <= 0) continue;
      revenue += drivers * c[4]; remaining -= drivers;
      assignments.push({zone:c[0], dow:c[1], hour:c[2], drivers, rph:c[4], capacity:cap});
    }
  }
  return {assignments, expected_revenue:revenue, drivers_placed:fleet-remaining,
          drivers_idle:remaining, cells_used:assignments.length,
          cells_considered:candidates.length};
}

if(typeof module !== "undefined") module.exports = {allocate, topK, capacityOf};
