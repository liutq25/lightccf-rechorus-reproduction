"""Audit the exact three author splits and their cached normalized graphs."""
from pathlib import Path
import hashlib
import json
import urllib.request
import numpy as np
import scipy.sparse as sp

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    rows,cols=[],[]
    by_user={}
    duplicates=0
    for line in path.read_text(encoding='utf-8').splitlines():
        values=[int(x) for x in line.split()]
        if not values:
            continue
        user,items=values[0],values[1:]
        if user in by_user:
            raise ValueError(f'Repeated user line: {path} {user}')
        by_user[user]=set(items)
        duplicates+=len(items)-len(set(items))
        rows.extend([user]*len(items))
        cols.extend(items)
    return np.asarray(rows),np.asarray(cols),by_user,duplicates


def main():
    output={}
    source_url='https://api.github.com/repos/ZzYUuuu/LightCCF/git/trees/45645ef15fc08e858ffc81e1b2530b4b1805ad34?recursive=1'
    with urllib.request.urlopen(source_url,timeout=30) as response:
        tree=json.load(response)
    upstream={row['path']:row['sha'] for row in tree['tree'] if row['type']=='blob'}
    all_valid=True
    for name in ('douban-book','tmall','amazon-book'):
        folder=ROOT/'vendor/LightCCF-main/dataset'/name
        train_u,train_i,train,duplicate_train=read(folder/'train.txt')
        test_u,test_i,test,duplicate_test=read(folder/'test.txt')
        n_users=int(max(train_u.max(),test_u.max()))+1
        n_items=int(max(train_i.max(),test_i.max()))+1
        du=np.bincount(train_u,minlength=n_users)
        di=np.bincount(train_i,minlength=n_items)
        weights=1/np.sqrt(du[train_u]*di[train_i])
        expected=sp.coo_matrix((np.r_[weights,weights],
                               (np.r_[train_u,train_i+n_users],np.r_[train_i+n_users,train_u])),
                              shape=(n_users+n_items,)*2).tocsr()
        cached=sp.load_npz(folder/'pre_Adj.npz').tocsr()
        assert cached.shape==expected.shape
        diff=cached-expected
        maximum=float(abs(diff.data).max()) if diff.nnz else 0.
        valid=maximum<1e-7
        all_valid=all_valid and valid
        overlaps=sum(len(items & train.get(u,set())) for u,items in test.items())
        assert overlaps==duplicate_train==duplicate_test==0
        raw=sp.coo_matrix((np.ones(len(train_u)),(train_u,train_i)),
                         shape=(n_users,n_items)).tocsr()
        target=sp.coo_matrix((np.ones(len(test_u)),(test_u,test_i)),
                            shape=(n_users,n_items)).tocsr()
        block=cached[:n_users,n_users:].copy()
        block.data[:]=1
        extra=block-block.multiply(raw)
        extra.eliminate_zeros()
        extra_pairs=list(zip(*(extra.tocoo().row.tolist(),extra.tocoo().col.tolist())))
        identities={}
        for p in (folder/'train.txt',folder/'test.txt',folder/'pre_Adj.npz'):
            data=p.read_bytes()
            git_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            remote=upstream.get(f'dataset/{name}/{p.name}')
            identities[p.name]=dict(git_blob_sha1=git_sha,pinned_remote_blob_sha1=remote,
                                     matches_pinned_author_source=git_sha==remote if remote else None)
            if remote:
                assert git_sha==remote,(name,p.name,'local differs from pinned upstream')
        clean=ROOT/'data/author_clean_graphs'/name/'pre_Adj.npz'
        clean.parent.mkdir(parents=True,exist_ok=True)
        sp.save_npz(clean,expected)
        check=sp.load_npz(clean)
        assert (check-expected).nnz==0
        output[name]=dict(users=n_users,items=n_items,
            train_interactions=len(train_u),test_interactions=len(test_u),
            total_interactions=len(train_u)+len(test_u),
            train_users=len(train),test_users=len(test),
            test_users_without_train=sum(u not in train for u in test),
            test_items_without_train=int(sum(di[test_i]==0)),
            duplicate_train=duplicate_train,duplicate_test=duplicate_test,train_test_overlap=overlaps,
            cache_nnz=int(cached.nnz),cache_max_abs_difference_from_train_only_graph=maximum,
            cache_matches_train_only_graph=valid,
            cache_missing_train_edges=int((raw-raw.multiply(block)).nnz),
            cache_edges_absent_from_train=len(extra_pairs),
            extra_pairs=[dict(user_id=int(u),item_id=int(i),is_test_positive=i in test.get(u,set()))
                         for u,i in extra_pairs],
            cache_test_edges=int(block.multiply(target).nnz),
            source_identity=identities,
            clean_graph_path=clean.relative_to(ROOT).as_posix(),
            clean_graph_sha256=hashlib.sha256(clean.read_bytes()).hexdigest(),
            sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (folder/'train.txt',folder/'test.txt',folder/'pre_Adj.npz')})
        print(name,n_users,n_items,len(train_u)+len(test_u),'cache error',maximum,
              'extra edges',len(extra_pairs),flush=True)
    (ROOT/'research/original_dataset_audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    if not all_valid:
        raise SystemExit('Published cache mismatch recorded; use audited train-only graph for clean experiments.')


if __name__=='__main__':
    main()
