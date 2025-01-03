import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import BlogList from './components/BlogList';
import BlogPost from './components/BlogPost';
import './styles.css';

const App = () => {
  return (
    <Router>
      <div className="app">
        <Switch>
          <Route exact path="/" component={BlogList} />
          <Route path="/post/:id" component={BlogPost} />
        </Switch>
      </div>
    </Router>
  );
};

export default App;
