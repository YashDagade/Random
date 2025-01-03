import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import firebase from '../firebase';
import './BlogList.css';

const BlogList = () => {
  const [posts, setPosts] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [tags, setTags] = useState([]);
  const [selectedTag, setSelectedTag] = useState('');
  const [sortByDate, setSortByDate] = useState(false);

  useEffect(() => {
    const fetchPosts = async () => {
      const postsCollection = await firebase.firestore().collection('posts').get();
      const postsData = postsCollection.docs.map(doc => ({ id: doc.id, ...doc.data() }));
      setPosts(postsData);
      const tagsData = [...new Set(postsData.flatMap(post => post.tags || []))];
      setTags(tagsData);
    };
    fetchPosts();
  }, []);

  const handleSearch = (e) => {
    setSearchTerm(e.target.value);
  };

  const handleTagSelect = (tag) => {
    setSelectedTag(tag);
  };

  const handleSortByDate = () => {
    setSortByDate(!sortByDate);
  };

  const filteredPosts = posts
    .filter(post => post.title.toLowerCase().includes(searchTerm.toLowerCase()))
    .filter(post => !selectedTag || (post.tags && post.tags.includes(selectedTag)))
    .sort((a, b) => sortByDate ? new Date(b.date) - new Date(a.date) : 0);

  return (
    <div className="blog-list">
      <div className="search-bar">
        <input
          type="text"
          placeholder="Search..."
          value={searchTerm}
          onChange={handleSearch}
        />
      </div>
      <div className="tags">
        {tags.map(tag => (
          <button
            key={tag}
            className={tag === selectedTag ? 'selected' : ''}
            onClick={() => handleTagSelect(tag)}
          >
            {tag}
          </button>
        ))}
      </div>
      <div className="sort-by-date">
        <button onClick={handleSortByDate}>
          {sortByDate ? 'Sort by Date (Descending)' : 'Sort by Date (Ascending)'}
        </button>
      </div>
      <div className="posts">
        {filteredPosts.map(post => (
          <div key={post.id} className="post">
            <Link to={`/post/${post.id}`}>
              <h2>{post.title}</h2>
              <p>{post.excerpt}</p>
            </Link>
          </div>
        ))}
      </div>
    </div>
  );
};

export default BlogList;
